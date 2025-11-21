use salsaa::protocol::execute;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .num_threads(64)
        .build_global()
        .unwrap();

    execute();
}
