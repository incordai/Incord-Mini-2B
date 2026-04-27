use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use qor4b::gemv::{forward_decode, mock::tiny_weights};
use qor4b::gemv_batch::{forward_decode_batch, SlotInferenceState};

fn bench_throughput(c: &mut Criterion) {
    let w = tiny_weights();

    let mut group = c.benchmark_group("decode");
    for n in [1usize, 4, 8, 16, 32, 64].iter() {
        group.throughput(Throughput::Elements(*n as u64));

        group.bench_with_input(BenchmarkId::new("batched", n), n, |b, &n| {
            let mut states: Vec<SlotInferenceState> = (0..n)
                .map(|_| SlotInferenceState::new(&w.config))
                .collect();
            // Init with non-zero token to avoid trivial paths
            for s in states.iter_mut() {
                s.last_token = 5;
            }
            b.iter(|| {
                let mut refs: Vec<&mut SlotInferenceState> = states.iter_mut().collect();
                let _ = forward_decode_batch(&w, &mut refs);
            });
        });

        group.bench_with_input(BenchmarkId::new("sequential", n), n, |b, &n| {
            let mut states: Vec<SlotInferenceState> = (0..n)
                .map(|_| SlotInferenceState::new(&w.config))
                .collect();
            for s in states.iter_mut() {
                s.last_token = 5;
            }
            b.iter(|| {
                for s in states.iter_mut() {
                    let _ = forward_decode(&w, s.last_token as usize, &mut s.cache);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
