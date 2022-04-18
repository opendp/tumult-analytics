# Prepare for release
- [ ] Identify features which must/must not be included in the release.
      Based on this, locate a [nightly pipeline](https://gitlab.com/tumult-labs/analytics/-/pipelines?page=1&scope=all&source=schedule) which meets these criteria.
- [ ] Determine an appropriate version number based on [semantic versioning](https://semver.org).

# Create release
- [ ] On the pipeline identified above, open the `trigger_release` job (do _not_ just trigger it from the pipeline view).
      This should give a view where custom variables may be set before triggering the job -- set one with the key `VERSION` and the value being whatever version number was decided on above, e.g. `1.3.2`.
      Launch the job, which will run for a while and kick off some new pipelines.
      Assuming the release tests pass, the package will be released without any further intervention.
- [ ] After half an hour or so, you will be assigned an MR from the release job.
      If there are no merge conflicts, this can be merged directly into `dev` without any additional action.
      If there are merge conflicts (most likely in the changelog), resolve them and then merge to `dev`.
      In either case, do not squash the commits.

# Handling failures
If the release tests fail, some manual action needs to be taken to get back to a releaseable state.
First, delete the tag of the failed release -- you will need to [un-protect tags](https://gitlab.com/tumult-labs/analytics/-/settings/repository#js-protected-tags-settings) to do this (there's an [open issue](https://gitlab.com/gitlab-org/gitlab/-/issues/20807) to allow an easier deletion flow for protected tags, but it hasn't seen attention in a long time).
Be sure to re-protect the tags when you're done, allowing the releaser token to create protected tags.
Having done that, you can then either:
- Fix the failure in `dev`, delete the release branch that was created, and restart the release process from a new nightly release that contains the fix.
- Apply the fix on the release branch and re-tag the fixed version with the same version number as the original tag.
