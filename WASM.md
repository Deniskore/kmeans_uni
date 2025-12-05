# Note on SIMD in WASM

The demo and `wasm` feature default to the scalar backend for portability. The `wide` crate can target wasm32 SIMD (`simd128`), but you must:

- Build with the `wide` feature enabled and `-C target-feature=+simd128`.
- Run in a browser/runtime that supports wasm SIMD.

The demo UI has a "Use SIMD" toggle, if you build without `simd128`/`wide` it will error.

## WASM Support

Builds a browser-friendly WASM package and runs a small K-Means demo.

## Prerequisites

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cargo install basic-http-server
```

## Build

The WASM demo lives in `examples/wasm`. The crate is built from the root with the `wasm` feature.

```bash
# from repo root
wasm-pack build --target web --out-dir examples/wasm/pkg -- --features wasm
```

To try SIMD, build with the SIMD toggle enabled in the demo and ensure both `wide` and `simd128` are available:

```bash
# enable the example's simd feature (for dep-wide) and pass simd128
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir examples/wasm/pkg -- --features wasm,wide
```

## Run locally

```bash
basic-http-server examples/wasm -a 127.0.0.1:8000
# then open http://127.0.0.1:8000/index.html
```

Click "Run demo" to train a small K-Means++ model on a random 2D dataset in the browser and see the predicted labels/centroids. The crate exposes a reusable wasm API (`kmeans_uni::wasm::WasmModel`) so other projects can depend on it directly, the demo re-exports the same class (constructor fits, accepts an iteration count, `predict` runs inference, `centroids` returns the trained centers).
