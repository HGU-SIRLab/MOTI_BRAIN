#!/usr/bin/env bash
# ZeroTier 기술 검증 — Tailscale이 이 네트워크에서 막혀서(docs/stage6_tailscale.md) 오버레이
# 방식 자체가 되는지부터 확인한다. 읽기만 하고 아무것도 바꾸지 않는다.
#
# 보는 것은 세 가지다:
#   1. ONLINE 인가 — 루트 서버에 닿았는가. Tailscale에서 막혔던 게 정확히 이 단계다
#   2. 네트워크 상태가 OK 인가 — 관리 콘솔에서 이 기기를 승인했는가
#   3. 🔴 DIRECT 인가 RELAY 인가 — Stage 6의 핵심 게이트. 중계면 그 지연이 음성 루프에 들어온다
#
# 사용:  bash scripts/check_zerotier.sh
set -u

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
no()   { printf '  \033[31m✗\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

# zerotier-cli는 보통 root가 필요하다. 토큰을 홈에 복사해두면 sudo 없이 된다:
#   sudo cp /var/lib/zerotier-one/authtoken.secret ~/.zeroTierOneAuthToken
#   sudo chown $USER ~/.zeroTierOneAuthToken && chmod 600 ~/.zeroTierOneAuthToken
ZT() { zerotier-cli "$@" 2>&1; }

say "1/4  설치·데몬"
command -v zerotier-cli >/dev/null 2>&1 || { no "zerotier-cli 미설치"; exit 1; }
systemctl is-active --quiet zerotier-one 2>/dev/null \
  && ok "zerotier-one 실행 중" || no "zerotier-one이 안 돈다 — sudo systemctl start zerotier-one"

INFO=$(ZT info)
if echo "$INFO" | grep -q "authtoken"; then
  no "권한 없음 — sudo로 돌리거나 authtoken을 홈에 복사할 것 (스크립트 주석 참고)"
  exit 1
fi

say "2/4  🔴 루트 서버에 닿았는가 (Tailscale이 막혔던 단계)"
echo "  $INFO"
if echo "$INFO" | grep -q ONLINE; then
  ok "ONLINE — 오버레이 조정 경로가 이 네트워크에서 열린다"
elif echo "$INFO" | grep -q OFFLINE; then
  no "OFFLINE — 루트 서버에 못 닿는다. ZeroTier도 같은 벽에 막혔다는 뜻이다."
  echo "     이러면 오버레이 방식 자체가 이 네트워크에서 안 되는 것이고,"
  echo "     남은 길은 IT 예외 요청이나 자체 호스팅(Headscale)이다."
  exit 1
else
  warn "상태 불명"
fi

say "3/4  네트워크·주소"
NETS=$(ZT listnetworks)
echo "$NETS" | sed 's/^/  /'
if echo "$NETS" | grep -q "OK"; then
  ok "네트워크 참여 완료"
elif echo "$NETS" | grep -q "ACCESS_DENIED"; then
  no "ACCESS_DENIED — my.zerotier.com에서 이 기기를 **승인(Auth 체크)** 해야 한다"
elif echo "$NETS" | grep -q "REQUESTING_CONFIGURATION"; then
  warn "설정 요청 중 — 승인 직후면 몇 초 기다렸다 다시"
else
  warn "참여한 네트워크가 없다 — sudo zerotier-cli join <네트워크ID>"
fi
ZT_IP=$(echo "$NETS" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/[0-9]+' | head -1 | cut -d/ -f1)
[ -n "$ZT_IP" ] && ok "이 기기의 ZeroTier 주소  $ZT_IP" || warn "아직 주소 미할당"

say "4/4  🔴 DIRECT 인가 RELAY 인가 (Stage 6의 핵심 게이트)"
PEERS=$(ZT peers)
echo "$PEERS" | head -12 | sed 's/^/  /'
LEAF=$(echo "$PEERS" | awk '$4=="LEAF"')
if [ -z "$LEAF" ]; then
  warn "상대 기기(LEAF)가 아직 안 보인다 — 로봇도 같은 네트워크에 넣고 승인할 것"
else
  if echo "$LEAF" | grep -qi "RELAY"; then
    no "RELAY — 중계를 타고 있다. 그 지연이 음성 루프에 그대로 들어온다(§18)"
  else
    ok "DIRECT — 중계 없이 붙었다"
  fi
  echo "$LEAF" | awk '{printf "     지연 %sms  경로 %s\n", $3, $6}'
fi

say "뇌 포트 (0.0.0.0 바인딩이라 오버레이 인터페이스 자동 포함)"
for port in 8765 8766; do
  ss -ltn 2>/dev/null | grep -qE "0\.0\.0\.0:$port|\[::\]:$port" \
    && ok ":$port 열림" || no ":$port 안 열림 — bash scripts/start_brain.sh"
done
[ -n "${ZT_IP:-}" ] && { say "로봇의 .env 에 넣을 값"; echo "  BRAIN_URI=ws://$ZT_IP:8765"; }
