#!/usr/bin/env bash
#
# Ensures a stable local code-signing identity ("meetre Local Signing") in the
# login keychain and prints its name. Idempotent.
#
# Why: ad-hoc signatures identify an app by the hash of that exact build, so
# every rebuild is a "new app" to macOS privacy (TCC) — Screen Recording /
# Microphone permissions granted to the previous build stop applying and the
# user is asked again (and capture fails until they re-allow + restart).
# Signed with a fixed certificate, the app's identity is "this bundle id +
# this certificate", which survives rebuilds, so one "Allow" sticks.
#
# The certificate is self-signed and local; it isn't trusted for anything
# else. Remove it any time in Keychain Access (search "meetre").
set -euo pipefail

NAME="meetre Local Signing"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-certificate -c "$NAME" "$KEYCHAIN" >/dev/null 2>&1; then
  echo "$NAME"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/req.cnf" <<EOF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $NAME
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
EOF
# macOS's LibreSSL: its PKCS#12 output is what `security import` reads.
/usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
  -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -config "$TMP/req.cnf" >/dev/null 2>&1
/usr/bin/openssl pkcs12 -export -inkey "$TMP/key.pem" -in "$TMP/cert.pem" \
  -name "$NAME" -out "$TMP/id.p12" -passout pass:meetre >/dev/null 2>&1
# -T: let codesign use the key without a keychain prompt on every build.
security import "$TMP/id.p12" -k "$KEYCHAIN" -P meetre -T /usr/bin/codesign >/dev/null
echo "$NAME"
