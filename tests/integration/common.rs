use qor4b::server::auth::admin::AdminState;
use qor4b::server::auth::middleware::AuthState;
use qor4b::server::auth::ratelimit::RateLimiter;
use qor4b::server::db::open;
use qor4b::server::http::router::build_router;
use redb::Database;
use std::sync::Arc;
use tempfile::TempDir;

#[allow(dead_code)] // db + server_secret held for ownership; accessed in M3 tests
pub struct TestServer {
    pub addr: std::net::SocketAddr,
    pub db: Arc<Database>,
    pub admin_token: String,
    pub server_secret: Arc<Vec<u8>>,
    handle: tokio::task::JoinHandle<()>,
    _dir: TempDir,
}

impl TestServer {
    pub async fn start() -> Self {
        let dir = TempDir::new().unwrap();
        let db = open(&dir.path().join("test.redb")).unwrap();
        let admin_token = "test-admin".to_string();
        let server_secret = Arc::new(b"test-secret".to_vec());

        let auth = Arc::new(AuthState {
            db: db.clone(),
            server_secret: server_secret.clone(),
            limiter: RateLimiter::new(),
        });
        let admin = Arc::new(AdminState {
            admin_token: admin_token.clone(),
        });
        let router = build_router(auth, admin);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router.into_make_service())
                .await
                .unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Self {
            addr,
            db,
            admin_token,
            server_secret,
            handle,
            _dir: dir,
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
