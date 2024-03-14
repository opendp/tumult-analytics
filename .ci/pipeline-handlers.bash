#!/bin/bash

function send_slack_webhook () {
    if [[ -z "$webhook_url" ]]; then
        echo "webhook_url unset"
        return 1
    fi
    if [[ -z "$message_content" ]]; then
        echo "message_content unset"
        return 1
    fi
    cat > body.json <<EOF
{
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "$message_content"
      }
    }
  ]
}
EOF
    echo "Request body:"
    cat body.json
    http_status=$(
      curl -XPOST -s -o response.json -w "%{http_code}" \
          "$webhook_url" \
          -H "Content-Type: application/json" -d "@body.json"
    )
    echo "Response body:"
    cat response.json
    if [[ $http_status -ne 200 ]]; then
      echo "\nGot unexpected HTTP status $http_status, exiting..."
      return 1
    fi
}

function nightly_handler () {
    set -euo pipefail

    if [[ -z "$NIGHTLY_HANDLER_TOKEN" ]]; then
        echo "NIGHTLY_HANDLER_TOKEN unset"
        exit 1
    fi
    if [[ -z "$NIGHTLY_SLACK_WEBHOOK_URL" ]]; then
        echo "NIGHTLY_SLACK_WEBHOOK_URL unset"
        exit 1
    fi

    # Grab information about the pipeline jobs from the GitLab API. Note that
    # the thing that the UI calls "trigger jobs" are called "bridges" in the
    # API, and are available through a different API call from the normal jobs.
    jobs_json="$(
      curl --header "Authorization: Bearer $NIGHTLY_HANDLER_TOKEN" \
          "$CI_API_V4_URL/projects/$CI_PROJECT_ID/pipelines/$CI_PIPELINE_ID/jobs"
    )"
    bridges_json="$(
      curl --header "Authorization: Bearer $NIGHTLY_HANDLER_TOKEN" \
          "$CI_API_V4_URL/projects/$CI_PROJECT_ID/pipelines/$CI_PIPELINE_ID/bridges"
    )"

    # Jobs/bridges in the handler stage may still be running, and should be
    # excluded from consideration by this handler.
    reported_jobs_json="$(
      echo $jobs_json | jq '[.[] | select(.stage != "handler")]'
    )"
    reported_bridges_json="$(
      echo $bridges_json | jq '[.[] | select(.stage != "handler")]'
    )"

    # Get lists of passed and failed jobs; for our purposes, any job whose
    # status is neither "success" nor "manual" (e.g. canceled jobs) has failed.
    passed_jobs_json="$(
      echo $reported_jobs_json | jq '[.[] | select(.status == "success")]'
    )"
    failed_jobs_json="$(
      echo $reported_jobs_json \
        | jq '[.[] | select(.status != "success" and .status != "manual")]'
    )"
    passed_bridges_json="$(
      echo $reported_bridges_json | jq '[.[] | select(.status == "success")]'
    )"
    failed_bridges_json="$(
      echo $reported_bridges_json \
        | jq '[.[] | select(.status != "success" and .status != "manual")]'
    )"
    echo -n "Passed jobs: "
    echo $passed_jobs_json | jq -r '[.[] | .name] | join(", ")'
    echo -n "Failed jobs: "
    echo $failed_jobs_json | jq -r '[.[] | .name] | join(", ")'
    echo -n "Passed bridges: "
    echo $passed_bridges_json | jq -r '[.[] | .name] | join(", ")'
    echo -n "Failed bridges: "
    echo $failed_bridges_json | jq -r '[.[] | .name] | join(", ")'
    echo

    if [[ "$(echo $failed_jobs_json | jq 'length == 0')" == 'true' ]]; then
      trigger_release_job_url="$(
        echo $jobs_json | jq -r '.[] | select(.name == "trigger_release") | .web_url'
      )"
      status_text="Nightly <$CI_PIPELINE_URL|pipeline> for \`$CI_COMMIT_BRANCH\`: :white_check_mark: Passed"
      body_text="Trigger a release with <$trigger_release_job_url|this job>"
      if [[ "$(echo $failed_bridges_json | jq 'length > 0')" == 'true' ]]; then
        bridge_links="$(
          echo $failed_bridges_json \
            | jq -r '[.[] | "<" + .downstream_pipeline.web_url + "|" + .name + ">"] | join(", ")'
        )"
        body_text="$body_text\nSome downstream pipelines are failing: $bridge_links"
        body_text="$body_text\nThis may be because of a bug, or just the result"
        body_text="$body_text of intentional backwards-incompatible changes."
      fi
    else
      job_links="$(
        echo "$failed_jobs_json" \
          | jq -r '[.[] | "<" + .web_url + "|" + .name + ">"] | join(", ")'
      )"
      status_text="Nightly <$CI_PIPELINE_URL|pipeline> for \`$CI_COMMIT_BRANCH\`: :x: Failed"
      body_text="Failed jobs: $job_links\n@channel"
    fi

    # Send the Slack notification
    webhook_url="$NIGHTLY_SLACK_WEBHOOK_URL"
    message_content="*$status_text*\n$body_text"
    send_slack_webhook
}

function release_handler () {
    set -euo pipefail

    if [[ -z "$RELEASE_SLACK_WEBHOOK_URL" ]]; then
        echo "RELEASE_SLACK_WEBHOOK_URL unset"
        exit 1
    fi

    links="<https://pypi.org/project/tmlt.analytics/|:package: Package Registry>"
    # Prereleases don't have docs published for them, so skip the docs link in
    # that case.
    if [[ ! "$CI_COMMIT_TAG" =~ ^[0-9]+\.[0-9]+\.[0-9]+-(alpha|beta|rc)\.[0-9]+$ ]]; then
        # Convert X.Y.Z semantic version to vX.Y for docs.
        docs_version="v$(echo $CI_COMMIT_TAG | sed -E 's/^([[:digit:]]+\.[[:digit:]]+).*/\1/')"
        links="$links    <https://docs.tmlt.dev/analytics/$docs_version|:page_facing_up: Docs>"
    fi
    links="$links    <$CI_PIPELINE_URL|:factory: Pipeline>"

    webhook_url="$RELEASE_SLACK_WEBHOOK_URL"
    message_content="*Analytics Release $CI_COMMIT_TAG*\n$links"
    send_slack_webhook
}
