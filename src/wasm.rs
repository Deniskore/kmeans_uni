#![cfg(all(feature = "wasm", target_arch = "wasm32"))]

use crate::{KMeans, KMeansBuilder, MetricType};
use cfg_if::cfg_if;
use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

macro_rules! wasm_model {
    ($name:ident, $float:ty, $to_array:ident, $doc:literal) => {
        #[wasm_bindgen]
        #[doc = $doc]
        pub struct $name {
            centroids: Vec<$float>,
            ncols: usize,
            k: usize,
        }

        #[wasm_bindgen]
        impl $name {
            /// Fit a model from a flat row-major point array of length `n_points * ncols`.
            /// Uses Euclidean distance and the provided `iterations` count (clamped to >= 1) for simplicity.
            #[wasm_bindgen(constructor)]
            pub fn new(
                points: Box<[$float]>,
                ncols: usize,
                k: usize,
                iterations: usize,
                use_simd: bool,
            ) -> Result<$name, JsValue> {
                if ncols == 0 {
                    return Err(JsValue::from_str("ncols must be > 0"));
                }
                if points.len() % ncols != 0 {
                    return Err(JsValue::from_str(
                        "points length must be divisible by ncols",
                    ));
                }

                let iterations = iterations.max(1);

                let base = KMeansBuilder::new(k).iterations(iterations).euclidean();

                cfg_if! {
                    if #[cfg(all(feature = "wide", target_feature = "simd128"))] {
                        let model = if use_simd {
                            base.cpu_simd()
                                .build()
                                .fit(points.as_ref(), ncols)
                                .map_err(to_js)?
                        } else {
                            base.cpu_scalar()
                                .build()
                                .fit(points.as_ref(), ncols)
                                .map_err(to_js)?
                        };
                    } else {
                        if use_simd {
                            return Err(JsValue::from_str("SIMD backend requested, but build is missing \"wide\" feature or simd128 target support"));
                        }
                        let model = base
                            .cpu_scalar()
                            .build()
                            .fit(points.as_ref(), ncols)
                            .map_err(to_js)?;
                    }
                }

                let centroids = model.centroids().to_vec();
                Ok($name {
                    centroids,
                    ncols: model.ncols(),
                    k: model.k(),
                })
            }

            /// Predict labels for the provided points (flat row-major array).
            pub fn predict(&self, points: Box<[$float]>) -> Result<Uint32Array, JsValue> {
                if points.len() % self.ncols != 0 {
                    return Err(JsValue::from_str(
                        "points length must be divisible by ncols",
                    ));
                }

                let model = KMeans::new(
                    self.centroids.clone(),
                    self.ncols,
                    self.k,
                    0.0,
                    MetricType::Euclidean,
                )
                .map_err(to_js)?;

                let labels = model.predict(points.as_ref()).map_err(to_js)?;
                let labels_u32: Vec<u32> = labels.iter().map(|&x| x as u32).collect();
                Ok(Uint32Array::from(labels_u32.as_slice()))
            }

            /// Return centroids as a flat Vec.
            pub fn centroids(&self) -> Vec<$float> {
                self.centroids.clone()
            }
        }
    };
}

wasm_model!(
    WasmModel,
    f32,
    to_array_f32,
    "Wasm-friendly K-Means++ model using f32 inputs/outputs."
);

wasm_model!(
    WasmModelF64,
    f64,
    to_array_f64,
    "Wasm-friendly K-Means++ model using f64 inputs/outputs."
);

fn to_js<E: core::fmt::Debug>(err: E) -> JsValue {
    JsValue::from_str(&format!("{err:?}"))
}
