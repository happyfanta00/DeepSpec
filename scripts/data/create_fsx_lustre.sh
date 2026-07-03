#!/usr/bin/env bash
#
# Create (and optionally mount) an FSx for Lustre file system in the SAME AZ as
# the EC2 instance this script runs on, then mount it locally.
#
# FSx for Lustre is single-AZ: the file system must live in one subnet, and only
# clients that can reach that subnet (same VPC / peered / etc.) can mount it. So
# the portable recipe is: discover this instance's region/AZ/subnet/VPC from the
# instance metadata service (IMDSv2), and build everything there. Run it on a
# box in us-east-1b and you get a us-east-1b file system; run it on a box in
# eu-west-1a and you get a eu-west-1a one -- no edits required.
#
# Usage:
#   ./scripts/data/create_fsx_lustre.sh              # create + mount
#   NO_MOUNT=1 ./scripts/data/create_fsx_lustre.sh   # create only
#
# Everything below is overridable via environment variables (defaults in []):
#   STORAGE_CAPACITY_GIB  [50400]  Rounded UP to a valid PERSISTENT_2 value.
#   PER_UNIT_THROUGHPUT   [250]    MB/s per TiB: 125|250|500|1000 (PERSISTENT_2).
#   DEPLOYMENT_TYPE       [PERSISTENT_2]
#   COMPRESSION           [NONE]   NONE|LZ4. LZ4 saves disk but barely helps on
#                                  fp8/bf16 hidden states (near-incompressible).
#   FS_NAME               [deepspec-target-cache]  Value of the Name tag.
#   MOUNT_POINT           [/mnt/fsx]
#   AWS_REGION / SUBNET_ID / SECURITY_GROUP_ID      Override IMDS discovery.
#
# COST WARNING: PERSISTENT_2 @ 250 MB/s/TiB over ~49 TiB provisions ~12 GB/s of
# baseline throughput and bills per GiB-month AND per MB/s/TiB. This is a large,
# ongoing charge. Delete when done (command printed at the end).

set -euo pipefail

# ---- tunables -------------------------------------------------------------
STORAGE_CAPACITY_GIB="${STORAGE_CAPACITY_GIB:-50400}"
PER_UNIT_THROUGHPUT="${PER_UNIT_THROUGHPUT:-250}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-PERSISTENT_2}"
COMPRESSION="${COMPRESSION:-NONE}"
FS_NAME="${FS_NAME:-deepspec-target-cache}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/fsx}"
SG_NAME="deepspec-fsx-lustre"

log()  { printf '\033[1;34m[fsx]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[fsx][warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fsx][error]\033[0m %s\n' "$*" >&2; exit 1; }

command -v aws >/dev/null || die "aws CLI not found."
command -v jq  >/dev/null || die "jq not found."

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ---- 0. discover placement from IMDSv2 ------------------------------------
imds() {
  curl -fsS -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" \
    "http://169.254.169.254/latest/meta-data/$1"
}

log "Querying instance metadata (IMDSv2)..."
IMDS_TOKEN="$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null)" \
  || die "Could not reach IMDS. Are you on an EC2 instance? For off-box use, set AWS_REGION and SUBNET_ID."

AWS_REGION="${AWS_REGION:-$(imds placement/region)}"
AZ="$(imds placement/availability-zone)"
MAC="$(imds mac)"
SUBNET_ID="${SUBNET_ID:-$(imds "network/interfaces/macs/${MAC}/subnet-id")}"
VPC_ID="$(imds "network/interfaces/macs/${MAC}/vpc-id")"
export AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION"

[ -n "$AWS_REGION" ] && [ -n "$SUBNET_ID" ] && [ -n "$VPC_ID" ] \
  || die "Failed to discover region/subnet/vpc from IMDS."

VPC_CIDR="$(aws ec2 describe-vpcs --vpc-ids "$VPC_ID" \
  --query 'Vpcs[0].CidrBlock' --output text)"

log "region=$AWS_REGION  az=$AZ  vpc=$VPC_ID ($VPC_CIDR)  subnet=$SUBNET_ID"

# ---- 1. validate / round storage capacity ---------------------------------
# PERSISTENT_2 SSD capacity must be 1200 GiB or a multiple of 2400 GiB.
round_capacity() {
  local want="$1"
  if [ "$want" -le 1200 ]; then echo 1200; return; fi
  local rem=$(( want % 2400 ))
  if [ "$rem" -eq 0 ]; then echo "$want"; else echo $(( want + 2400 - rem )); fi
}
ORIG_CAP="$STORAGE_CAPACITY_GIB"
STORAGE_CAPACITY_GIB="$(round_capacity "$STORAGE_CAPACITY_GIB")"
if [ "$STORAGE_CAPACITY_GIB" != "$ORIG_CAP" ]; then
  warn "Rounded StorageCapacity ${ORIG_CAP} -> ${STORAGE_CAPACITY_GIB} GiB (must be 1200 or a multiple of 2400)."
fi
log "StorageCapacity=${STORAGE_CAPACITY_GIB} GiB (~$(( STORAGE_CAPACITY_GIB / 1024 )) TiB), \
throughput=${PER_UNIT_THROUGHPUT} MB/s/TiB, type=${DEPLOYMENT_TYPE}, compression=${COMPRESSION}"

# ---- 2. service-linked role (best effort) ---------------------------------
if ! aws iam get-role --role-name AWSServiceRoleForAmazonFSx >/dev/null 2>&1; then
  log "Creating service-linked role for FSx (ignored if it already exists)..."
  aws iam create-service-linked-role --aws-service-name fsx.amazonaws.com >/dev/null 2>&1 \
    || warn "Could not create service-linked role; FSx may auto-create it, or you may lack iam:CreateServiceLinkedRole."
fi

# ---- 3. security group (idempotent) ---------------------------------------
# FSx for Lustre needs inbound TCP 988 and 1018-1023 from mounting clients.
SG_ID="${SECURITY_GROUP_ID:-}"
if [ -z "$SG_ID" ]; then
  SG_ID="$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=${VPC_ID}" "Name=group-name,Values=${SG_NAME}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
  if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    log "Creating security group ${SG_NAME}..."
    SG_ID="$(aws ec2 create-security-group \
      --group-name "$SG_NAME" \
      --description "Lustre traffic for FSx (created by create_fsx_lustre.sh)" \
      --vpc-id "$VPC_ID" \
      --query 'GroupId' --output text)"
    for PORTS in "988" "1018-1023"; do
      FROM="${PORTS%%-*}"; TO="${PORTS##*-}"
      aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --ip-permissions "IpProtocol=tcp,FromPort=${FROM},ToPort=${TO},IpRanges=[{CidrIp=${VPC_CIDR},Description=lustre}]" \
        >/dev/null
    done
  fi
fi
log "SecurityGroup=$SG_ID"

# ---- 4. create the file system --------------------------------------------
LUSTRE_CONF="$(jq -nc \
  --arg dt "$DEPLOYMENT_TYPE" \
  --arg comp "$COMPRESSION" \
  --argjson tput "$PER_UNIT_THROUGHPUT" \
  '{DeploymentType:$dt, PerUnitStorageThroughput:$tput, DataCompressionType:$comp}')"

log "Calling CreateFileSystem... (this provisions billable capacity)"
FS_ID="$(aws fsx create-file-system \
  --file-system-type LUSTRE \
  --storage-type SSD \
  --storage-capacity "$STORAGE_CAPACITY_GIB" \
  --subnet-ids "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --lustre-configuration "$LUSTRE_CONF" \
  --tags "Key=Name,Value=${FS_NAME}" \
  --query 'FileSystem.FileSystemId' --output text)"
log "Created ${FS_ID}. Waiting for it to become AVAILABLE (typically 5-15 min)..."

# ---- 5. wait for AVAILABLE (FSx has no built-in CLI waiter) ----------------
while true; do
  STATUS="$(aws fsx describe-file-systems --file-system-ids "$FS_ID" \
    --query 'FileSystems[0].Lifecycle' --output text)"
  case "$STATUS" in
    AVAILABLE) log "File system is AVAILABLE."; break ;;
    CREATING)  printf '.' >&2; sleep 20 ;;
    *) REASON="$(aws fsx describe-file-systems --file-system-ids "$FS_ID" \
         --query 'FileSystems[0].FailureDetails.Message' --output text 2>/dev/null || true)"
       die "Unexpected lifecycle '${STATUS}'. ${REASON}" ;;
  esac
done
printf '\n' >&2

DNS_NAME="$(aws fsx describe-file-systems --file-system-ids "$FS_ID" \
  --query 'FileSystems[0].DNSName' --output text)"
MOUNT_NAME="$(aws fsx describe-file-systems --file-system-ids "$FS_ID" \
  --query 'FileSystems[0].LustreConfiguration.MountName' --output text)"

log "DNSName=$DNS_NAME  MountName=$MOUNT_NAME"

# ---- 6. mount (unless NO_MOUNT) -------------------------------------------
MOUNT_CMD="${SUDO} mount -t lustre -o relatime,flock ${DNS_NAME}@tcp:/${MOUNT_NAME} ${MOUNT_POINT}"
if [ "${NO_MOUNT:-0}" = "1" ]; then
  log "NO_MOUNT=1 set; skipping mount. To mount later:"
  echo "  ${SUDO} mkdir -p ${MOUNT_POINT} && ${MOUNT_CMD}"
else
  log "Mounting at ${MOUNT_POINT}..."
  $SUDO mkdir -p "$MOUNT_POINT"
  eval "$MOUNT_CMD"
  $SUDO chown "$(id -u):$(id -g)" "$MOUNT_POINT" 2>/dev/null || true
  log "Mounted:"
  df -hT "$MOUNT_POINT" >&2
fi

# ---- 7. summary ------------------------------------------------------------
cat >&2 <<EOF

============================================================
 FSx for Lustre ready
   FileSystemId : ${FS_ID}
   Region / AZ  : ${AWS_REGION} / ${AZ}
   Capacity     : ${STORAGE_CAPACITY_GIB} GiB  @ ${PER_UNIT_THROUGHPUT} MB/s/TiB
   Mount point  : ${MOUNT_POINT}

 Re-mount later (e.g. after reboot):
   ${SUDO} mkdir -p ${MOUNT_POINT}
   ${MOUNT_CMD}

 Point target-cache output at it:
   --output-dir ${MOUNT_POINT}/qwen3_4b_target_cache

 DELETE when finished (stops billing):
   aws fsx delete-file-system --file-system-id ${FS_ID} --region ${AWS_REGION}
============================================================
EOF
