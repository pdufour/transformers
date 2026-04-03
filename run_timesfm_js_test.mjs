/**
 * TimesFM 2.5 ONNX via Transformers.js — same contract as Hugging Face Python:
 * `past_values` is a **sequence** of 1D series (one array/tensor per row), not a
 * single flat 2D layout. ONNX export: `past_values` -> mean_predictions, full_predictions.
 *
 * The Flax ONNX from export_onnx_flax.py (inputs/masks) is not compatible with this path.
 */
import { pipeline } from '../transformers.js/packages/transformers/src/transformers.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function runTestSuite() {
    try {
        const model_path = path.resolve(__dirname, './onnx');
        console.log(`\nStarting test suite — model: ${model_path}\n`);

        const forecaster = await pipeline('time-series-forecasting', model_path);
        const horizon = forecaster.model.config.horizon_length;

        const testCases = [
            { name: 'Single Series (Batch=1)', batch: 1, seq: 512 },
            { name: 'Batch Processing (Batch=2)', batch: 2, seq: 512 },
            { name: 'Batch Processing (Batch=4)', batch: 4, seq: 512 },
            { name: 'Batch Processing (Batch=8)', batch: 8, seq: 512 },
        ];

        let passed = 0;
        for (const tc of testCases) {
            console.log(`[TEST] ${tc.name}: ${tc.batch} series × length ${tc.seq} (sequence input)`);

            /** @type {number[][]} Sequence of 1D series — same shape as `Sequence[torch.Tensor]` in Python. */
            const past_values = [];
            for (let b = 0; b < tc.batch; b++) {
                past_values.push(
                    Array.from({ length: tc.seq }, (_, i) => Math.sin(i / 310)),
                );
            }

            const inputData = { past_values };

            try {
                const start = performance.now();
                const output = await forecaster(inputData);
                const end = performance.now();

                const mShape = [...output.mean_predictions.dims];
                const fShape = [...output.full_predictions.dims];

                if (mShape[0] !== tc.batch || mShape[1] !== horizon) {
                    throw new Error(
                        `Mean shape mismatch: expected [${tc.batch}, ${horizon}], got [${mShape}]`,
                    );
                }

                console.log(
                    `  Passed in ${((end - start) / 1000).toFixed(3)}s — Mean [${mShape}], Full [${fShape}]`,
                );
                passed++;
            } catch (err) {
                console.log(`  Failed: ${err.message}`);
            }
        }

        console.log(`\nResults: ${passed}/${testCases.length} tests passed.\n`);
        if (passed === testCases.length) {
            console.log('All tests passed.');
        }
    } catch (e) {
        console.error('Test suite failed to initialize:', e);
        process.exitCode = 1;
    }
}

runTestSuite();
