#!/usr/bin/env python
"""Various CI pipeline handlers."""

import requests
import sys
import os
import logging
from http import HTTPStatus
import argparse

_log = logging.getLogger()


def _send_slack_webhook(url: str, content: str):
    """Create a Slack post with the given content using a webhook."""
    body = {
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": content}}]
    }

    _log.info("Sending Slack webhook")
    _log.debug(f"Body: {body}")
    resp = requests.post(url, json=body)
    if resp.status_code > 399:
        _log.critical(
            f"Slack webhook failed (HTTP {resp.status_code}), response: {resp.text}"
        )
        raise RuntimeError("Error sending webhook")

    _log.info(f"Slack webhook sent, status {resp.status_code}")


def _gitlab_api_get(path: str, token: str):
    """Query the GitLab API at the given path, using the given token for auth."""
    api_v4_url = os.environ["CI_API_V4_URL"]
    headers = {"Authorization": f"Bearer {token}"}
    return requests.get(f"{api_v4_url}/{path}", headers=headers)


def _get_pipeline_jobs(project_id: int, pipeline_id: int, token: str):
    resp = _gitlab_api_get(
        f"projects/{project_id}/pipelines/{pipeline_id}/jobs", token=token
    )
    if resp.status_code > 399:
        _log.critical(
            f"Unable to list pipeline {pipeline_id} jobs (HTTP {resp.status_code}), "
            f"response: {resp.text}"
        )
        raise RuntimeError("Error making GitLab API requests")
    return resp


def _nightly_handler(args):
    token = os.environ["NIGHTLY_HANDLER_TOKEN"]
    webhook_url = os.environ["NIGHTLY_SLACK_WEBHOOK_URL"]
    project_id = int(os.environ["CI_PROJECT_ID"])
    pipeline_id = int(os.environ["CI_PIPELINE_ID"])
    pipeline_url = os.environ["CI_PIPELINE_URL"]
    pipeline_branch = os.environ["CI_COMMIT_BRANCH"]

    _log.info(f"Handling pipeline {pipeline_id} as nightly pipeline...")

    jobs_json = _get_pipeline_jobs(project_id, pipeline_id, token).json()
    jobs = {j["name"]: j for j in jobs_json}

    bridges_resp = _gitlab_api_get(
        f"projects/{project_id}/pipelines/{pipeline_id}/bridges", token=token
    )
    if bridges_resp.status_code > 399:
        _log.critical(
            f"Unable to list downstream pipelines (HTTP {bridges_resp.status_code}), "
            f"response: {bridges_resp.text}"
        )
        raise RuntimeError("Error making GitLab API requests")
    bridges_json = bridges_resp.json()
    for bridge in bridges_json:
        downstream_pipeline = bridge["downstream_pipeline"]
        bridge_jobs_json = _get_pipeline_jobs(
            downstream_pipeline["project_id"], downstream_pipeline["id"], token
        ).json()
        for job in bridge_jobs_json:
            jobs[f"{bridge['name']}:{job['name']}"] = job

    jobs = {j.replace(" ", ""): status for j, status in jobs.items()}
    # The handler job is obviously still running when this script runs, which
    # makes the script consider it as failed. So, we ignore it.
    jobs.pop("nightly_handler")

    unknown_allow_failures = set(args.allow_failure) - set(jobs.keys())
    if unknown_allow_failures:
        _log.warning(
            "Jobs were set to allow failure, but they do not exist: "
            + " ".join(unknown_allow_failures)
        )

    passed_jobs = {j: body for j, body in jobs.items() if body["status"] == "success"}
    failed_jobs = {
        j: body
        for j, body in jobs.items()
        if body["status"] not in {"success", "manual"} and j not in args.allow_failure
    }
    allowed_failed_jobs = {
        j: body
        for j, body in jobs.items()
        if body["status"] not in {"success", "manual"} and j in args.allow_failure
    }

    def format_job_status(jobs):
        return [f"{j} ({b['status']})" for j, b in jobs.items()]

    _log.info(f"Passed jobs: {', '.join(passed_jobs.keys())}")
    _log.info(f"Failed jobs: {', '.join(format_job_status(failed_jobs))}")
    _log.info(
        f"Allowed failed jobs: {', '.join(format_job_status(allowed_failed_jobs))}"
    )

    if failed_jobs:
        status_text = (
            f"Nightly <{pipeline_url}|pipeline> for `{pipeline_branch}`: :x: Failed"
        )
        job_links = [f"<{body['web_url']}|{j}>" for j, body in failed_jobs.items()]
        body_text = f"Failed jobs: {', '.join(job_links)}\n@channel"
    else:
        try:
            trigger_release_url = next(
                j["web_url"] for j in jobs if j["name"] == "trigger_release"
            )
        except StopIteration:
            raise RuntimeError("No trigger_release job found")
        status_text = (
            f"Nightly <{pipeline_url}|pipeline> for `{pipeline_branch}`: "
            ":white_check_mark: Passed"
        )
        body_text = f"Trigger a release with <{trigger_release_url}|this job>"

        if allowed_failed_jobs:
            job_links = [f"<{body['web_url']}|{j}>" for j, body in failed_jobs.items()]
            body_text += (
                "\nSome jobs which are allowed to fail are failing: "
                + ", ".join(job_links)
                + "\nThis may be because of a bug, or just the result of intentional "
                "backwards-incompatible changes. If you are making a release based on "
                "this commit, ensure that we know the cause of all of these failures "
                "and that they are expected."
            )

    _send_slack_webhook(webhook_url, f"*{status_text}*\n{body_text}")


def _release_handler(args):
    raise NotImplementedError(
        "Continue using the Bash version of the release handler for now"
    )


def _arg_parser_nightly_handler(subparsers):
    sp = subparsers.add_parser("nightly", help="Run the nightly pipeline handler")
    sp.add_argument("--allow-failure", action="append", default=[])
    sp.set_defaults(func=_nightly_handler)


def _arg_parser_release_handler(subparsers):
    sp = subparsers.add_parser("release", help="Run the release handler")
    sp.set_defaults(func=_release_handler)


def _arg_parser():
    parser = argparse.ArgumentParser(prog="pipeline-handler")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    _arg_parser_nightly_handler(subparsers)
    _arg_parser_release_handler(subparsers)

    return parser


def main(argv):
    parser = _arg_parser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    _log.info(f"Arguments: {args}")

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
