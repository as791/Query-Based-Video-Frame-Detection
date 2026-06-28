#!/bin/sh
set -eu

REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
STREAM="${AUTOSCALER_STREAM:-pipeline.events}"
POLL_INTERVAL_SEC="${AUTOSCALER_POLL_INTERVAL_SEC:-15}"
SCALE_COOLDOWN_SEC="${AUTOSCALER_SCALE_COOLDOWN_SEC:-60}"
COMPOSE_FILE="${AUTOSCALER_COMPOSE_FILE:-/workspace/compose/docker-compose.yml}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-compose}"
DRY_RUN="${AUTOSCALER_DRY_RUN:-false}"

LAST_SCALE_FILE="/tmp/autoscaler-last-scale"
mkdir -p "$LAST_SCALE_FILE"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

redis_value_for_group() {
  group="$1"
  key="$2"
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --raw XINFO GROUPS "$STREAM" 2>/dev/null | awk -v group="$group" -v wanted="$key" '
    $0 == "name" {
      getline value
      active = (value == group)
      next
    }
    active && $0 == wanted {
      getline value
      print value
      exit
    }
  '
}

number_or_zero() {
  value="${1:-0}"
  case "$value" in
    ""|"-"|"(nil)") printf '0' ;;
    *[!0-9]*) printf '0' ;;
    *) printf '%s' "$value" ;;
  esac
}

group_backlog() {
  group="$1"
  pending="$(number_or_zero "$(redis_value_for_group "$group" pending)")"
  lag="$(number_or_zero "$(redis_value_for_group "$group" lag)")"
  printf '%s' "$((pending + lag))"
}

running_replicas() {
  service="$1"
  count="$(docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null | wc -l | tr -d ' ')"
  number_or_zero "$count"
}

desired_replicas() {
  backlog="$1"
  current="$2"
  min="$3"
  max="$4"
  step="$5"
  up_at="$6"
  down_below="$7"

  desired="$current"
  if [ "$desired" -lt "$min" ]; then
    desired="$min"
  fi

  if [ "$backlog" -ge "$up_at" ]; then
    increments="$(( ((backlog - up_at) / up_at) + 1 ))"
    desired="$((desired + (increments * step)))"
  elif [ "$backlog" -le "$down_below" ] && [ "$current" -gt "$min" ]; then
    desired="$((current - step))"
  fi

  if [ "$desired" -lt "$min" ]; then desired="$min"; fi
  if [ "$desired" -gt "$max" ]; then desired="$max"; fi
  printf '%s' "$desired"
}

cooldown_ready() {
  service="$1"
  marker="$LAST_SCALE_FILE/$service"
  now="$(date +%s)"
  last=0
  if [ -f "$marker" ]; then
    last="$(cat "$marker" 2>/dev/null || printf '0')"
  fi
  [ "$((now - last))" -ge "$SCALE_COOLDOWN_SEC" ]
}

mark_scaled() {
  service="$1"
  date +%s > "$LAST_SCALE_FILE/$service"
}

scale_service() {
  service="$1"
  desired="$2"
  current="$3"
  backlog="$4"

  if [ "$desired" -eq "$current" ]; then
    log "service=$service backlog=$backlog replicas=$current unchanged"
    return
  fi

  if ! cooldown_ready "$service"; then
    log "service=$service backlog=$backlog replicas=$current desired=$desired cooldown"
    return
  fi

  log "service=$service backlog=$backlog scaling $current -> $desired"
  if [ "$DRY_RUN" != "true" ]; then
    docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" up -d --scale "$service=$desired" "$service"
  fi
  mark_scaled "$service"
}

evaluate_service() {
  service="$1"
  group="$2"
  min="$3"
  max="$4"
  step="$5"
  up_at="$6"
  down_below="$7"

  backlog="$(group_backlog "$group")"
  current="$(running_replicas "$service")"
  desired="$(desired_replicas "$backlog" "$current" "$min" "$max" "$step" "$up_at" "$down_below")"
  scale_service "$service" "$desired" "$current" "$backlog"
}

wait_for_redis() {
  until redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" PING >/dev/null 2>&1; do
    log "waiting for redis at $REDIS_HOST:$REDIS_PORT"
    sleep 2
  done
}

log "starting autoscaler stream=$STREAM project=$COMPOSE_PROJECT_NAME dry_run=$DRY_RUN"
wait_for_redis

while :; do
  evaluate_service \
    "${CHUNKER_SERVICE:-chunker}" \
    "${CHUNKER_GROUP:-cg-chunker}" \
    "${CHUNKER_MIN_REPLICAS:-1}" \
    "${CHUNKER_MAX_REPLICAS:-4}" \
    "${CHUNKER_SCALE_STEP:-1}" \
    "${CHUNKER_SCALE_UP_AT:-50}" \
    "${CHUNKER_SCALE_DOWN_BELOW:-5}"

  evaluate_service \
    "${NORMALIZER_SERVICE:-normalizer}" \
    "${NORMALIZER_GROUP:-cg-normalizer}" \
    "${NORMALIZER_MIN_REPLICAS:-1}" \
    "${NORMALIZER_MAX_REPLICAS:-4}" \
    "${NORMALIZER_SCALE_STEP:-1}" \
    "${NORMALIZER_SCALE_UP_AT:-50}" \
    "${NORMALIZER_SCALE_DOWN_BELOW:-5}"

  evaluate_service \
    "${EXTRACTOR_SERVICE:-extractor}" \
    "${EXTRACTOR_GROUP:-cg-extractor}" \
    "${EXTRACTOR_MIN_REPLICAS:-1}" \
    "${EXTRACTOR_MAX_REPLICAS:-8}" \
    "${EXTRACTOR_SCALE_STEP:-1}" \
    "${EXTRACTOR_SCALE_UP_AT:-25}" \
    "${EXTRACTOR_SCALE_DOWN_BELOW:-3}"

  sleep "$POLL_INTERVAL_SEC"
done
