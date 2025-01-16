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

function release_handler () {
    set -euo pipefail

    if [[ -z "$RELEASE_SLACK_WEBHOOK_URL" ]]; then
        echo "RELEASE_SLACK_WEBHOOK_URL unset"
        exit 1
    fi

    links="<https://pypi.org/project/tmlt.analytics/|:package: Package Registry>"
    links="$links    <$CI_PIPELINE_URL|:factory: Pipeline>"

    webhook_url="$RELEASE_SLACK_WEBHOOK_URL"
    message_content="*Analytics Release $CI_COMMIT_TAG*\n$links"
    send_slack_webhook
}
