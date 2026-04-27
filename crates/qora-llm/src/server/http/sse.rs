use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::{Stream, StreamExt};
use std::convert::Infallible;
use std::time::Duration;

pub fn into_sse_response<S>(stream: S) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: Stream<Item = Event> + Send + 'static,
{
    Sse::new(stream.map(Ok::<_, Infallible>)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text(": keepalive"),
    )
}
