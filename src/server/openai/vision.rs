//! Parse OpenAI's image_url field — supports data: URLs (inline base64) and http(s):// URLs.

use base64::{engine::general_purpose::STANDARD as B64, Engine};

#[derive(Debug, Clone)]
pub enum VisionContent {
    /// Inline image bytes from a data: URL.
    Inline { mime: String, bytes: Vec<u8> },
    /// Remote URL — caller fetches via reqwest.
    Remote { url: String },
}

pub fn parse_image_url(url: &str) -> anyhow::Result<VisionContent> {
    if let Some(rest) = url.strip_prefix("data:") {
        // data:[<mime>][;base64],<data>
        let (header, data) = rest.split_once(',')
            .ok_or_else(|| anyhow::anyhow!("invalid data url: missing comma"))?;
        let mime = header.split(';').next().unwrap_or("application/octet-stream").to_string();
        let bytes = if header.contains("base64") {
            B64.decode(data)?
        } else {
            // Fallback: percent-encoded text
            urlencoding::decode(data)?.as_bytes().to_vec()
        };
        return Ok(VisionContent::Inline { mime, bytes });
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(VisionContent::Remote { url: url.to_string() });
    }
    Err(anyhow::anyhow!("unsupported image_url scheme: {}", &url[..url.len().min(40)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_data_url_base64() {
        // "hello" base64 = "aGVsbG8="
        let url = "data:image/png;base64,aGVsbG8=";
        match parse_image_url(url).unwrap() {
            VisionContent::Inline { mime, bytes } => {
                assert_eq!(mime, "image/png");
                assert_eq!(bytes, b"hello");
            }
            _ => panic!("expected inline"),
        }
    }

    #[test]
    fn parses_https() {
        match parse_image_url("https://example.com/x.png").unwrap() {
            VisionContent::Remote { url } => assert!(url.starts_with("https://")),
            _ => panic!("expected remote"),
        }
    }

    #[test]
    fn parses_http() {
        match parse_image_url("http://example.com/x.png").unwrap() {
            VisionContent::Remote { url } => assert!(url.starts_with("http://")),
            _ => panic!("expected remote"),
        }
    }

    #[test]
    fn rejects_unknown_scheme() {
        assert!(parse_image_url("ftp://x.png").is_err());
    }

    #[test]
    fn rejects_malformed_data_url() {
        // No comma after data: prefix
        assert!(parse_image_url("data:image/png;base64").is_err());
    }
}
