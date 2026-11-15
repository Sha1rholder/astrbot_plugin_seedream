[简体中文](.astrbot-plugin/i18n/README.zh-CN.md)

# Seedream 5.5 Image Generation Adapter for AstrBot Agents

An AstrBot plugin for generating and editing images via Seedream 5.5 through the Volcengine API. Designed with minimal prompts and reasonable defaults.

Requires an API key in the plugin settings or the `VOLCENGINE_API_KEY` environment variable.

Generates one PNG in the background. Keeps request and image archives, with quotas, proxies, and storage cleanup.

Agents can use up to 10 reference images from attachments, local paths, or HTTPS URLs, and choose transparent backgrounds.

Each request directory stores `request.yaml` and the provider response in `response.yaml`. The plugin reads `data[0].url` from the saved response, downloads it to `0.png`, and delivers the image. The response remains available if download or delivery fails. Reuse its URL within 24 hours of generation, or the local PNG afterward while the archive exists.
