#![cfg(feature = "mock-weights")]

use qor4b::server::auth::admin::AdminState;
use qor4b::server::auth::middleware::AuthState;
use qor4b::server::auth::ratelimit::RateLimiter;
use qor4b::server::db::open;
use qor4b::server::http::handlers::chat::ChatState;
use qor4b::server::http::router::build_router;
use qor4b::server::scheduler::queue::Queue;
use qor4b::server::usage::tracker::UsageTracker;
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

        let queue = Queue::new(10);
        let weights = Arc::new(qor4b::gemv::mock::tiny_weights());
        let tokenizer = build_test_tokenizer();
        let usage = UsageTracker::new(db.clone());

        let pool = qor4b::server::scheduler::pool::SlotPool::new(8); // small slot pool for tests

        let batcher = qor4b::server::scheduler::batcher::Batcher {
            pool: pool.clone(),
            queue: queue.clone(),
            weights: weights.clone(),
            tokenizer: tokenizer.clone(),
        };
        std::thread::Builder::new()
            .name("test-batcher".into())
            .spawn(move || batcher.run())
            .expect("test batcher spawn");

        let chat_state = Arc::new(ChatState {
            queue,
            weights,
            tokenizer,
            usage,
        });

        let router = build_router(auth, admin, chat_state);

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

/// Construct a tokenizer for tests. Looks for tokenizer.json at well-known paths.
fn build_test_tokenizer() -> Arc<qor4b::tokenizer::QoraTokenizer> {
    let candidates = ["model/tokenizer.json", "tests/fixtures/tokenizer.json"];
    for path in candidates {
        if std::path::Path::new(path).exists()
            && let Ok(t) = qor4b::tokenizer::QoraTokenizer::from_file(std::path::Path::new(path))
        {
            return Arc::new(t);
        }
    }
    panic!("integration tests need a tokenizer.json file; expected at one of: {candidates:?}");
}
