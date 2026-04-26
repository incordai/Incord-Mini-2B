use std::net::SocketAddr;
use qor4b::server::http::router::build_router;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

pub struct TestServer {
    pub addr: SocketAddr,
    handle: JoinHandle<()>,
}

impl TestServer {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = build_router();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router.into_make_service()).await.unwrap();
        });
        // Give the server a moment to be ready
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Self { addr, handle }
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
