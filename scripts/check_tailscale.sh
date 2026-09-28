#!/usr/bin/env bash
# Stage 6 확인 — tailnet이 제대로 섰는지, 그리고 direct인지.
#
# `tailscale status` 출력을 눈으로 판정하기 번거로워서 만들었다. 핵심은 direct 여부다 —
# relay(DERP)로 떨어지면 중계 서버를 한 번 더 거치고 그 지연이 음성 루프에 그대로 들어온다
# (스펙 §18, Q15). 읽기만 하고 아무것도 바꾸지 않는다.
#
# 사용:  bash scripts/check_tailscale.sh [로봇_호스트명]
set -u

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
no()   { printf '  \033[31m✗\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

PEER="${1:-}"

say "1/5  설치·데몬"
if ! command -v tailscale >/dev/null 2>&1; then
  no "tailscale 미설치 — docs/stage6_tailscale.md §1"; exit 1
fi
ok "$(tailscale version | head -1)"
if systemctl is-active --quiet tailscaled 2>/dev/null; then ok "tailscaled 실행 중"
else no "tailscaled가 안 돈다 — sudo systemctl start tailscaled"; fi

say "2/5  로그인·주소"
BACKEND=$(tailscale status --json 2>/dev/null | grep -oP '"BackendState":\s*"\K[^"]+' | head -1)
case "$BACKEND" in
  Running) ok "로그인됨 (Running)" ;;
  NeedsLogin|Stopped) no "로그인 안 됨 ($BACKEND) — sudo tailscale up"; exit 1 ;;
  *) warn "상태 불명 ($BACKEND)" ;;
esac
TS_IP=$(tailscale ip -4 2>/dev/null | head -1)
DNS=$(tailscale status --json 2>/dev/null | grep -oP '"DNSName":\s*"\K[^"]+' | head -1)
DNS=${DNS%.}
[ -n "$TS_IP" ] && ok "tailnet IP  $TS_IP" || no "tailnet IP 없음"
if [ -n "$DNS" ]; then ok "MagicDNS    $DNS"
else warn "MagicDNS 이름 없음 — 관리 콘솔에서 MagicDNS를 켜면 IP 대신 이름을 쓸 수 있다"; fi

say "3/5  뇌 포트가 tailnet에서 열려 있나"
# 0.0.0.0 바인딩이라 자동으로 포함돼야 한다. 확인만 한다.
for port in 8765 8766; do
  if ss -ltn 2>/dev/null | grep -qE "0\.0\.0\.0:$port|\[::\]:$port"; then
    ok ":$port 열림 (모든 인터페이스 — tailnet 포함)"
  else
    no ":$port 안 열림 — bash scripts/start_brain.sh"
  fi
done

say "4/5  피어"
tailscale status 2>/dev/null | sed 's/^/  /' | head -12
if [ -z "$PEER" ]; then
  warn "로봇 호스트명을 인자로 주면 direct 여부까지 확인한다:  bash $0 <로봇이름>"
  exit 0
fi

say "5/5  🔴 direct 인가 (Stage 6의 핵심 게이트)"
OUT=$(timeout 25 tailscale ping --timeout 3s -c 5 "$PEER" 2>&1)
echo "$OUT" | sed 's/^/  /'
if echo "$OUT" | grep -q "via DERP"; then
  no "relay(DERP)를 타고 있다 — 성능 저하 모드다."
  echo "     중계 서버를 한 번 더 거치므로 그 지연이 음성 루프에 그대로 들어온다."
  echo "     양쪽 NAT에 달린 문제라 네트워크를 바꿔가며 다시 볼 것 (스펙 §18, Q15)."
elif echo "$OUT" | grep -qE "via [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:"; then
  ok "direct — 중계 없이 붙었다"
  RTT=$(echo "$OUT" | grep -oP 'in \K[0-9]+ms' | tail -1)
  [ -n "$RTT" ] && echo "     왕복 $RTT (같은 LAN 기준선과 비교: 1ms 내외)"
else
  warn "판정 불가 — 위 출력을 그대로 보고할 것"
fi

say "로봇의 .env 에 넣을 값"
TARGET=${DNS:-$TS_IP}
echo "  BRAIN_URI=ws://$TARGET:8765"
[ -n "$DNS" ] && echo "  (IP가 아니라 MagicDNS 이름을 쓴다 — 주소가 바뀌어도 따라간다)"
