# AstrBot Seedream 5.5适配器

AstrBot插件，使用Seedream 5.5，通过火山引擎API支持文生图和图片编辑，保持极简提示词和合理默认值

需要在插件配置中填写API密钥，或通过环境变量`VOLCENGINE_API_KEY`提供

后台生成单张PNG。保留请求和图片存档，支持配额、代理和存储清理

Agent可使用最多10张来自附件、本地路径或HTTPS链接的参考图，并选择透明背景

每个请求目录先保存API响应到`response.yaml`，再解析其中的`data[0].url`下载PNG并无损转换为`0.webp`并自动发送。下载或投递失败仍保留响应。生成后24小时内可复用URL，之后可使用尚未被清理的本地WebP
