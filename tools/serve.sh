#!/usr/bin/env bash
#
# Install what the site needs, then serve it locally with live reload.
#
#   tools/serve.sh
#
# Safe to re-run: the install steps are skipped once they have been done.
#
# Environment overrides:
#   HOST=0.0.0.0        bind address           (default 127.0.0.1)
#   PORT=4000           site port              (default 4000)
#   LIVERELOAD_PORT=... reload socket port     (default 35729)
#   POLL=1              use --force_polling, for editing over a network mount

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-4000}"
LIVERELOAD_PORT="${LIVERELOAD_PORT:-35729}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

# The Gemfile wants Jekyll 4.4, which needs Ruby 3.1 or newer. Ubuntu 24.04's
# ruby-full is 3.2, so the distro package is enough and there is no need for
# rbenv or rvm.
ruby_new_enough() {
  command -v ruby >/dev/null 2>&1 || return 1
  ruby -e 'exit(Gem::Version.new(RUBY_VERSION) >= Gem::Version.new("3.1") ? 0 : 1)' \
    >/dev/null 2>&1
}

if ruby_new_enough; then
  echo "==> ruby $(ruby -e 'print RUBY_VERSION') already installed"
else
  echo "==> installing ruby and build tools (apt)"
  # build-essential, ruby-dev and zlib1g-dev are not optional: --livereload
  # pulls in eventmachine, which has a C extension that is compiled on install.
  $SUDO apt-get update
  $SUDO apt-get install -y ruby-full ruby-dev build-essential zlib1g-dev
fi

if ! command -v bundle >/dev/null 2>&1; then
  echo "==> installing bundler"
  $SUDO gem install bundler --no-document
fi

# Gems go in vendor/, which is both gitignored and in the `exclude` list in
# _config.yml, so Jekyll will not try to publish them as site content.
bundle config set --local path vendor/bundle

if bundle check >/dev/null 2>&1; then
  echo "==> gems already installed"
else
  echo "==> bundle install"
  bundle install
fi

JEKYLL_ARGS=(
  serve
  --livereload
  --host "$HOST"
  --port "$PORT"
  --livereload-port "$LIVERELOAD_PORT"
)

if [[ -n "${POLL:-}" ]]; then
  JEKYLL_ARGS+=(--force_polling)
fi

cat <<EOF

==> serving on http://$HOST:$PORT/scale-your-amd/

    The trailing /scale-your-amd/ is required; \`/\` on its own returns 404,
    because baseurl in _config.yml is part of every URL.

    Over SSH, forward both ports:
      ssh -L $PORT:localhost:$PORT -L $LIVERELOAD_PORT:localhost:$LIVERELOAD_PORT <host>

    Edits to _config.yml are the one thing not picked up automatically.
    Restart after those.

EOF

exec bundle exec jekyll "${JEKYLL_ARGS[@]}"
