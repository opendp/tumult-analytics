function injectBanner(content) {
  var body = document.getElementsByClassName('bd-content')[0];
  if (body) {
    body.prepend(content);
  } else {
    console.warn("Unable to find body element, skipping banner injection");
  }
}

function init() {
  const banner_config_url = DOCUMENTATION_OPTIONS.URL_ROOT + "banner-config.json";
  fetch(banner_config_url)
    .then((resp) => {
      if (resp.status != 200) {
        console.warn(
          "Unable to fetch banner configuration, got status code " + resp.status
        );
        return;
      }
      return resp.json();
    }).then((config) => {
      if (config.content != null) {
        var banner = document.createElement("div");
        banner.innerHTML = config.content;
        banner.className = "tmlt-banner-warning";
        injectBanner(banner);
      } else {
        console.log("Banner config has no content, not inserting banner")
      }
    }).catch((err) => console.log(err));
}

$(document).ready(function () {
  init();
});
