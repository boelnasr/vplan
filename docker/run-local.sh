#!/usr/bin/env bash
# R8 local check (docs/plan.md W3.4): build the vplan image, then inside it
#   (a) run the full test suite, and
#   (b) prepare 10 witness-validated tasks through the real pipeline (heuristic plan, verify, execute, judge).
# Nothing here needs credentials: no env file is mounted or passed, and both containers run without a network.
#
# Usage: docker/run-local.sh   (from any directory; needs Docker Engine and membership in the docker group)
# Output: runs/r8-local/<UTC stamp>/{build.log, image.json, pytest.log, prepare.log, cohort/}
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${VPLAN_IMAGE:-vplan:local}"
out="${root}/runs/r8-local/$(date -u +%Y%m%dT%H%M%SZ)"

if ! docker info > /dev/null 2>&1; then
    echo "error: cannot reach the Docker daemon. Is it running, and is $(id -un) in the docker group?" \
         "(A new login shell picks up a new group.)" >&2
    exit 2
fi
mkdir -p "${out}"

sha="$(git -C "${root}" rev-parse HEAD 2> /dev/null || true)"
revision=unknown
git_env=()
if [[ -n "${sha}" ]]; then
    dirty=false
    revision="${sha}"
    if [[ -n "$(git -C "${root}" status --porcelain 2> /dev/null)" ]]; then
        dirty=true
        revision="${sha}-dirty"
    fi
    # The image has neither git nor .git, so manifests made inside it take the host checkout's SHA from these
    # (vplan.evaluate.gitinfo, source "env"); eval run refuses to start without one unless --allow-no-git.
    git_env=(--env "VPLAN_GIT_SHA=${sha}" --env "VPLAN_GIT_DIRTY=${dirty}")
fi

echo "== build ${image} (revision ${revision})"
build_start=${SECONDS}
docker build --progress=plain --file "${root}/docker/Dockerfile" --build-arg "VPLAN_GIT_SHA=${revision}" \
    --tag "${image}" "${root}" 2>&1 | tee "${out}/build.log"
build_s=$(( SECONDS - build_start ))
docker image inspect "${image}" > "${out}/image.json"
# The local image ID, a content digest (an image built here has no registry digest until it is pushed).
# The W3.1 runner records it in its manifest (spec §14).
digest="$(docker image inspect --format '{{.Id}}' "${image}")"

status=0
echo "== (a) full test suite in the container"
docker run --rm --network none --env "VPLAN_IMAGE_DIGEST=${digest}" ${git_env[@]+"${git_env[@]}"} \
    --entrypoint python "${image}" \
    -m pytest -q -p no:cacheprovider 2>&1 | tee "${out}/pytest.log" || status=1

echo "== (b) 10 tasks through the real pipeline, no keys"
# Runs as the caller's uid/gid: the output stays owned by the caller, and on a native Linux host the image's own
# user could not write to the caller's directory.
docker run --rm --network none ${git_env[@]+"${git_env[@]}"} --user "$(id -u):$(id -g)" \
    --env "VPLAN_IMAGE_DIGEST=${digest}" --volume "${out}:/out" "${image}" \
    pilot prepare --output /out/cohort --n 10 --seed 0 --max-candidates 50 2>&1 | tee "${out}/prepare.log" \
    || status=1

size_mb=$(( $(docker image inspect --format '{{.Size}}' "${image}") / 1000000 ))
echo "== ${image} ${digest}: built in ${build_s} s, ${size_mb} MB"
echo "suite: $(tail -n 1 "${out}/pytest.log")"
echo "tasks: $(tail -n 2 "${out}/prepare.log" | paste -sd ' ' -)"
echo "output: ${out}"
exit "${status}"
