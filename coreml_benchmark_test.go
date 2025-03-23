package onnxruntime_go

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"testing"
	"time"
)

// Example represents a single tokenized example for ONNX model input
type Example struct {
	InputIDs      []int64 `json:"input_ids"`
	AttentionMask []int64 `json:"attention_mask"`
	TokenTypeIDs  []int64 `json:"token_type_ids"`
}

// BatchExample represents a batch of tokenized examples for ONNX model input
type BatchExample struct {
	BatchSize     int       `json:"batch_size"`
	InputIDs      [][]int64 `json:"input_ids"`
	AttentionMask [][]int64 `json:"attention_mask"`
	TokenTypeIDs  [][]int64 `json:"token_type_ids"`
}

// ComplexData represents the structure of our JSON data with different example types
type ComplexData struct {
	Regular   []Example      `json:"regular"`
	Long      []Example      `json:"long"`
	MaxLength []Example      `json:"max_length"`
	Batched   []BatchExample `json:"batched"`
}

// initializeRuntimeWithVerboseLogs initializes the ONNX runtime with verbose level logging
// This allows us to see detailed CoreML execution information
func initializeRuntimeWithVerboseLogs(t testing.TB) {
	if IsInitialized() {
		return
	}
	SetSharedLibraryPath(getSharedLibraryPathForBenchmarks(t))
	e := InitializeEnvironment(WithLogLevelWarning())
	if e != nil {
		t.Fatalf("Failed setting up onnxruntime environment with warning logs: %s\n", e)
	}
}

// cleanupRuntime destroys the ONNX runtime environment
func cleanupRuntime(t testing.TB) {
	e := DestroyEnvironment()
	if e != nil {
		t.Fatalf("Error cleaning up environment: %s\n", e)
	}
}

// getSharedLibraryPathForBenchmarks is copied from onnxruntime_test.go to make this package self-contained
func getSharedLibraryPathForBenchmarks(t testing.TB) string {
	toReturn := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if toReturn != "" {
		return toReturn
	}
	if runtime.GOOS == "windows" {
		return "test_data/onnxruntime.dll"
	}
	if runtime.GOARCH == "arm64" {
		if runtime.GOOS == "darwin" {
			return "test_data/onnxruntime_arm64.dylib"
		}
		return "test_data/onnxruntime_arm64.so"
	}
	if runtime.GOARCH == "amd64" && runtime.GOOS == "darwin" {
		return "test_data/onnxruntime_amd64.dylib"
	}
	return "test_data/onnxruntime.so"
}

// TestMiniLM_OptimalConfiguration demonstrates the best configuration for MiniLM models
func TestMiniLM_OptimalConfiguration(t *testing.T) {
	initializeRuntimeWithVerboseLogs(t)
	defer cleanupRuntime(t)

	// Test with the fastest model (FP16) and optimal compute settings (CPUOnly)
	modelPath := "all-MiniLM-L6-v2/onnx/model_fp16.onnx"
	computeUnits := "CPUOnly"
	modelFormat := "MLProgram"

	// Typical input sizes for embedding use cases
	testCases := []struct {
		batchSize int
		seqLength int
	}{
		{1, 32},  // Single short sentence
		{1, 128}, // Single paragraph
		{4, 32},  // Small batch of sentences
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("Batch%d_Seq%d", tc.batchSize, tc.seqLength), func(t *testing.T) {
			t.Logf("Testing batch=%d, sequence=%d", tc.batchSize, tc.seqLength)

			// Create input tensors
			inputIds := make([]int64, tc.batchSize*tc.seqLength)
			attentionMask := make([]int64, tc.batchSize*tc.seqLength)
			tokenTypeIds := make([]int64, tc.batchSize*tc.seqLength)

			// Fill with dummy values
			for i := 0; i < tc.batchSize*tc.seqLength; i++ {
				inputIds[i] = int64(i % 1000)
				attentionMask[i] = 1
				tokenTypeIds[i] = 0
			}

			// Create tensors
			inputIdsTensor, err := NewTensor(NewShape(int64(tc.batchSize), int64(tc.seqLength)), inputIds)
			if err != nil {
				t.Fatalf("Error creating input_ids tensor: %v", err)
			}
			defer inputIdsTensor.Destroy()

			attentionMaskTensor, err := NewTensor(NewShape(int64(tc.batchSize), int64(tc.seqLength)), attentionMask)
			if err != nil {
				t.Fatalf("Error creating attention_mask tensor: %v", err)
			}
			defer attentionMaskTensor.Destroy()

			tokenTypeIdsTensor, err := NewTensor(NewShape(int64(tc.batchSize), int64(tc.seqLength)), tokenTypeIds)
			if err != nil {
				t.Fatalf("Error creating token_type_ids tensor: %v", err)
			}
			defer tokenTypeIdsTensor.Destroy()

			// Create output tensor
			outputShape := NewShape(int64(tc.batchSize), int64(tc.seqLength), 384)
			outputData := make([]float32, tc.batchSize*tc.seqLength*384)
			outputTensor, err := NewTensor(outputShape, outputData)
			if err != nil {
				t.Fatalf("Error creating output tensor: %v", err)
			}
			defer outputTensor.Destroy()

			// Create session options
			sessionOptions, err := NewSessionOptions()
			if err != nil {
				t.Fatalf("Error creating session options: %v", err)
			}
			defer sessionOptions.Destroy()

			// Configure CoreML provider with optimal settings
			coreMLOptions := NewCoreMLProviderOptions()
			coreMLOptions.SetModelFormat(CoreMLModelFormat(modelFormat))
			coreMLOptions.SetMLComputeUnits(CoreMLComputeUnits(computeUnits))
			coreMLOptions.SetRequireStaticInputShapes(true)
			coreMLOptions.SetEnableOnSubgraphs(true)
			coreMLOptions.SetModelCacheDirectory("/tmp")
			coreMLOptions.SetAllowLowPrecisionAccumulationOnGPU(true)
			err = sessionOptions.AppendExecutionProviderCoreMLV2(coreMLOptions)
			if err != nil {
				t.Fatalf("Error configuring CoreML provider: %v", err)
			}

			// Create session with model
			startTime := time.Now()
			session, err := NewAdvancedSession(
				modelPath,
				[]string{"input_ids", "attention_mask", "token_type_ids"},
				[]string{"last_hidden_state"},
				[]Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor},
				[]Value{outputTensor},
				sessionOptions,
			)
			if err != nil {
				t.Fatalf("Error creating session: %v", err)
			}
			defer session.Destroy()

			// Report model load time
			loadTime := time.Since(startTime)
			t.Logf("Model load time: %v", loadTime)

			// First run (with compilation)
			startTime = time.Now()
			err = session.Run()
			if err != nil {
				t.Fatalf("Error on first inference: %v", err)
			}
			firstRunTime := time.Since(startTime)
			t.Logf("First run time (with compilation): %v", firstRunTime)

			// Multiple runs for accurate timing
			const numRuns = 10
			startTime = time.Now()
			for i := 0; i < numRuns; i++ {
				err = session.Run()
				if err != nil {
					t.Fatalf("Error on run %d: %v", i+1, err)
				}
			}
			totalTime := time.Since(startTime)

			// Calculate metrics
			avgTime := totalTime / time.Duration(numRuns)
			tokensPerSecond := float64(tc.batchSize*tc.seqLength*numRuns) / totalTime.Seconds()
			totalTokens := tc.batchSize * tc.seqLength

			// Report performance
			t.Logf("Average inference time: %v", avgTime)
			t.Logf("Tokens per batch: %d", totalTokens)
			t.Logf("Tokens/second: %.2f", tokensPerSecond)

			// Memory usage for model
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			t.Logf("Current memory usage: %d MB", m.Alloc/1024/1024)
		})
	}
}

// TestMiniLM_RealData loads and benchmarks tokenized data from the JSON file
func TestMiniLM_RealData(t *testing.T) {
	initializeRuntimeWithVerboseLogs(t)
	defer cleanupRuntime(t)

	// Read the tokenized data from JSON
	jsonData, err := os.ReadFile("tokenized_data.json")
	if err != nil {
		t.Fatalf("Error reading tokenized data: %v", err)
	}

	// Parse JSON data - the file contains an array of examples
	type Example struct {
		InputIDs      []int64 `json:"input_ids"`
		AttentionMask []int64 `json:"attention_mask"`
		TokenTypeIDs  []int64 `json:"token_type_ids"`
	}
	var examples []Example

	if err := json.Unmarshal(jsonData, &examples); err != nil {
		t.Fatalf("Error parsing JSON: %v", err)
	}

	// Validate data
	if len(examples) == 0 {
		t.Fatalf("No examples found in tokenized data")
	}

	numExamples := len(examples)
	t.Logf("Loaded %d examples of tokenized data", numExamples)

	// Calculate total tokens and find max sequence length
	totalTokens := 0
	maxSeqLen := 0
	for _, example := range examples {
		seqLen := len(example.InputIDs)
		totalTokens += seqLen
		if seqLen > maxSeqLen {
			maxSeqLen = seqLen
		}
	}
	t.Logf("Total tokens across all examples: %d", totalTokens)
	t.Logf("Maximum sequence length: %d", maxSeqLen)

	// Use the optimal configuration
	modelPath := "all-MiniLM-L6-v2/onnx/model_fp16.onnx"

	// Create session options with CoreML provider
	sessionOptions, err := NewSessionOptions()
	if err != nil {
		t.Fatalf("Error creating session options: %v", err)
	}
	defer sessionOptions.Destroy()

	// Configure CoreML provider with optimal settings
	coreMLOptions := NewCoreMLProviderOptions()
	coreMLOptions.SetModelFormat("MLProgram")
	coreMLOptions.SetMLComputeUnits("CPUOnly")
	coreMLOptions.SetEnableOnSubgraphs(true)
	coreMLOptions.SetModelCacheDirectory("/tmp")
	err = sessionOptions.AppendExecutionProviderCoreMLV2(coreMLOptions)
	if err != nil {
		t.Fatalf("Error configuring CoreML provider: %v", err)
	}

	// Benchmark 10 randomly selected examples (or all if less than 10)
	numExamplesToTest := 10
	if numExamplesToTest > numExamples {
		numExamplesToTest = numExamples
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	totalInferenceTime := time.Duration(0)
	var firstRunTime time.Duration
	isFirstRun := true

	// Run inference on random examples
	for i := 0; i < numExamplesToTest; i++ {
		exampleIdx := rng.Intn(numExamples)
		example := examples[exampleIdx]

		// Get sequence length for this example
		seqLen := len(example.InputIDs)

		// Pad inputs to fixed size if needed
		paddedInputIDs := make([]int64, maxSeqLen)
		paddedAttentionMask := make([]int64, maxSeqLen)
		paddedTokenTypeIDs := make([]int64, maxSeqLen)

		// Copy actual data and set attention mask
		copy(paddedInputIDs, example.InputIDs)
		copy(paddedAttentionMask, example.AttentionMask)
		copy(paddedTokenTypeIDs, example.TokenTypeIDs)

		// Create input tensors with batch size of 1 and max sequence length
		inputIdsTensor, err := NewTensor(NewShape(1, int64(maxSeqLen)), paddedInputIDs)
		if err != nil {
			t.Fatalf("Error creating input_ids tensor: %v", err)
		}
		defer inputIdsTensor.Destroy()

		attentionMaskTensor, err := NewTensor(NewShape(1, int64(maxSeqLen)), paddedAttentionMask)
		if err != nil {
			t.Fatalf("Error creating attention_mask tensor: %v", err)
		}
		defer attentionMaskTensor.Destroy()

		tokenTypeIdsTensor, err := NewTensor(NewShape(1, int64(maxSeqLen)), paddedTokenTypeIDs)
		if err != nil {
			t.Fatalf("Error creating token_type_ids tensor: %v", err)
		}
		defer tokenTypeIdsTensor.Destroy()

		// Create output tensor (assuming 384-dim embeddings per token)
		outputShape := NewShape(1, int64(maxSeqLen), 384)
		outputData := make([]float32, maxSeqLen*384)
		outputTensor, err := NewTensor(outputShape, outputData)
		if err != nil {
			t.Fatalf("Error creating output tensor: %v", err)
		}
		defer outputTensor.Destroy()

		// Create session with model
		startTime := time.Now()
		session, err := NewAdvancedSession(
			modelPath,
			[]string{"input_ids", "attention_mask", "token_type_ids"},
			[]string{"last_hidden_state"},
			[]Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor},
			[]Value{outputTensor},
			sessionOptions,
		)
		if err != nil {
			t.Fatalf("Error creating session: %v", err)
		}

		// Time the model load separately on first example
		if i == 0 {
			loadTime := time.Since(startTime)
			t.Logf("Model load time: %.4f seconds", loadTime.Seconds())
		}

		// Run inference
		startInference := time.Now()
		err = session.Run()
		if err != nil {
			t.Fatalf("Error running inference: %v", err)
		}
		inferenceTime := time.Since(startInference)

		// Track first run time separately (compilation happens first time)
		if isFirstRun {
			firstRunTime = inferenceTime
			isFirstRun = false
		} else {
			totalInferenceTime += inferenceTime
		}

		// Log performance metrics - only count non-padding tokens for speed
		tokensPerSecond := float64(seqLen) / inferenceTime.Seconds()
		t.Logf("Example %d: %d tokens in %v (%.2f tokens/sec)",
			exampleIdx, seqLen, inferenceTime, tokensPerSecond)

		// Cleanup session after each run
		session.Destroy()
	}

	// Calculate and report stats
	avgInferenceTime := totalInferenceTime / time.Duration(numExamplesToTest-1) // exclude first run
	t.Logf("First run time: %.4f ms", float64(firstRunTime.Microseconds())/1000.0)
	t.Logf("Average inference time (after first run): %.4f ms", float64(avgInferenceTime.Microseconds())/1000.0)
	t.Logf("Average tokens/second: %.2f", float64(totalTokens)/totalInferenceTime.Seconds())

	t.Logf("BEST CONFIGURATION FOR MINILM MODELS:")
	t.Logf("- FP16 model (model_fp16.onnx): Faster load times and good accuracy")
	t.Logf("- 'CPUOnly' compute setting: Faster compilation and inference")
	t.Logf("- 'MLProgram' model format: Best compatibility")
}

// TestModelFormatComparison compares performance of different model formats with and without CoreML
func TestModelFormatComparison(t *testing.T) {
	initializeRuntimeWithVerboseLogs(t)
	defer cleanupRuntime(t)

	// Read the tokenized data from JSON
	jsonData, err := os.ReadFile("tokenized_data.json")
	if err != nil {
		t.Fatalf("Error reading tokenized data: %v", err)
	}

	// Parse JSON data
	type Example struct {
		InputIDs      []int64 `json:"input_ids"`
		AttentionMask []int64 `json:"attention_mask"`
		TokenTypeIDs  []int64 `json:"token_type_ids"`
	}
	var examples []Example

	if err := json.Unmarshal(jsonData, &examples); err != nil {
		t.Fatalf("Error parsing JSON: %v", err)
	}

	// Validate data
	if len(examples) == 0 {
		t.Fatalf("No examples found in tokenized data")
	}

	numExamples := len(examples)
	t.Logf("Loaded %d examples of tokenized data", numExamples)

	// Calculate max sequence length
	maxSeqLen := 0
	for _, example := range examples {
		seqLen := len(example.InputIDs)
		if seqLen > maxSeqLen {
			maxSeqLen = seqLen
		}
	}
	t.Logf("Maximum sequence length: %d", maxSeqLen)

	// Define test cases
	testCases := []struct {
		name      string
		modelPath string
		useCoreML bool
	}{
		{"FP32-CoreML", "all-MiniLM-L6-v2/onnx/model.onnx", true},
		{"FP16-CoreML", "all-MiniLM-L6-v2/onnx/model_fp16.onnx", true},
		{"INT8-CoreML", "all-MiniLM-L6-v2/onnx/model_int8.onnx", true},
		{"Quantized-CoreML", "all-MiniLM-L6-v2/onnx/model_quantized.onnx", true},
		{"FP32-CPU", "all-MiniLM-L6-v2/onnx/model.onnx", false},
		{"FP16-CPU", "all-MiniLM-L6-v2/onnx/model_fp16.onnx", false},
		{"INT8-CPU", "all-MiniLM-L6-v2/onnx/model_int8.onnx", false},
		{"Quantized-CPU", "all-MiniLM-L6-v2/onnx/model_quantized.onnx", false},
	}

	// Select a fixed set of examples (same for all tests)
	exampleIndices := []int{0, 100, 500, 1000, 2000}
	if len(exampleIndices) > numExamples {
		exampleIndices = exampleIndices[:numExamples]
	}

	// Run all test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create session options
			sessionOptions, err := NewSessionOptions()
			if err != nil {
				t.Fatalf("Error creating session options: %v", err)
			}
			defer sessionOptions.Destroy()

			// Configure with CoreML if specified
			if tc.useCoreML {
				coreMLOptions := NewCoreMLProviderOptions()
				coreMLOptions.SetModelFormat("MLProgram")
				coreMLOptions.SetMLComputeUnits("CPUOnly")
				coreMLOptions.SetEnableOnSubgraphs(true)
				coreMLOptions.SetModelCacheDirectory("/tmp")
				err = sessionOptions.AppendExecutionProviderCoreMLV2(coreMLOptions)
				if err != nil {
					t.Fatalf("Error configuring CoreML provider: %v", err)
				}
			}

			totalInferenceTime := time.Duration(0)
			var firstRunTime time.Duration
			isFirstRun := true

			// Run inference on selected examples
			for i, exampleIdx := range exampleIndices {
				example := examples[exampleIdx]

				// Get sequence length for this example
				seqLen := len(example.InputIDs)

				// Pad inputs to fixed size if needed
				paddedInputIDs := make([]int64, maxSeqLen)
				paddedAttentionMask := make([]int64, maxSeqLen)
				paddedTokenTypeIDs := make([]int64, maxSeqLen)

				// Copy actual data and set attention mask
				copy(paddedInputIDs, example.InputIDs)
				copy(paddedAttentionMask, example.AttentionMask)
				copy(paddedTokenTypeIDs, example.TokenTypeIDs)

				// Create input tensors with batch size of 1 and max sequence length
				inputIdsTensor, err := NewTensor(NewShape(1, int64(maxSeqLen)), paddedInputIDs)
				if err != nil {
					t.Fatalf("Error creating input_ids tensor: %v", err)
				}
				defer inputIdsTensor.Destroy()

				attentionMaskTensor, err := NewTensor(NewShape(1, int64(maxSeqLen)), paddedAttentionMask)
				if err != nil {
					t.Fatalf("Error creating attention_mask tensor: %v", err)
				}
				defer attentionMaskTensor.Destroy()

				tokenTypeIdsTensor, err := NewTensor(NewShape(1, int64(maxSeqLen)), paddedTokenTypeIDs)
				if err != nil {
					t.Fatalf("Error creating token_type_ids tensor: %v", err)
				}
				defer tokenTypeIdsTensor.Destroy()

				// Create output tensor (assuming 384-dim embeddings per token)
				outputShape := NewShape(1, int64(maxSeqLen), 384)
				outputData := make([]float32, maxSeqLen*384)
				outputTensor, err := NewTensor(outputShape, outputData)
				if err != nil {
					t.Fatalf("Error creating output tensor: %v", err)
				}
				defer outputTensor.Destroy()

				// Create session with model
				startTime := time.Now()
				session, err := NewAdvancedSession(
					tc.modelPath,
					[]string{"input_ids", "attention_mask", "token_type_ids"},
					[]string{"last_hidden_state"},
					[]Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor},
					[]Value{outputTensor},
					sessionOptions,
				)
				if err != nil {
					t.Fatalf("Error creating session: %v", err)
				}

				// Time the model load separately on first example
				if i == 0 {
					loadTime := time.Since(startTime)
					t.Logf("Model load time: %.4f seconds", loadTime.Seconds())
				}

				// Run inference
				startInference := time.Now()
				err = session.Run()
				if err != nil {
					t.Fatalf("Error running inference: %v", err)
				}
				inferenceTime := time.Since(startInference)

				// Track first run time separately (compilation happens first time)
				if isFirstRun {
					firstRunTime = inferenceTime
					isFirstRun = false
				} else {
					totalInferenceTime += inferenceTime
				}

				// Log performance metrics - only count non-padding tokens for speed
				tokensPerSecond := float64(seqLen) / inferenceTime.Seconds()
				t.Logf("Example %d: %d tokens in %v (%.2f tokens/sec)",
					exampleIdx, seqLen, inferenceTime, tokensPerSecond)

				// Cleanup session after each run
				session.Destroy()
			}

			// Calculate and report stats
			avgInferenceTime := totalInferenceTime / time.Duration(len(exampleIndices)-1) // exclude first run
			t.Logf("First run time: %.4f ms", float64(firstRunTime.Microseconds())/1000.0)
			t.Logf("Average inference time (after first run): %.4f ms", float64(avgInferenceTime.Microseconds())/1000.0)

			// Calculate average tokens per second
			totalTokens := 0
			for _, idx := range exampleIndices[1:] { // Skip first example for avg calculation
				totalTokens += len(examples[idx].InputIDs)
			}
			tokensPerSecond := float64(totalTokens) / totalInferenceTime.Seconds()
			t.Logf("Average tokens/second: %.2f", tokensPerSecond)
		})
	}
}

// TestLongSequenceComparison tests performance with much longer sequences and batching
func TestLongSequenceComparison(t *testing.T) {
	initializeRuntimeWithVerboseLogs(t)
	defer cleanupRuntime(t)

	// Read the tokenized data from JSON
	jsonData, err := os.ReadFile("tokenized_data.json")
	if err != nil {
		t.Fatalf("Error reading tokenized data: %v", err)
	}

	// Parse JSON data
	type Example struct {
		InputIDs      []int64 `json:"input_ids"`
		AttentionMask []int64 `json:"attention_mask"`
		TokenTypeIDs  []int64 `json:"token_type_ids"`
	}
	var examples []Example

	if err := json.Unmarshal(jsonData, &examples); err != nil {
		t.Fatalf("Error parsing JSON: %v", err)
	}

	// Validate data
	if len(examples) == 0 {
		t.Fatalf("No examples found in tokenized data")
	}

	numExamples := len(examples)
	t.Logf("Loaded %d examples of tokenized data", numExamples)

	// Create long sequence tests
	testCases := []struct {
		name               string
		modelPath          string
		useCoreML          bool
		batchSize          int
		sequenceMultiplier int
	}{
		{"Long-FP32-CPU-SingleBatch", "all-MiniLM-L6-v2/onnx/model.onnx", false, 1, 10},
		{"Long-FP32-CoreML-SingleBatch", "all-MiniLM-L6-v2/onnx/model.onnx", true, 1, 10},
		{"Long-INT8-CPU-SingleBatch", "all-MiniLM-L6-v2/onnx/model_int8.onnx", false, 1, 10},
		{"Long-INT8-CoreML-SingleBatch", "all-MiniLM-L6-v2/onnx/model_int8.onnx", true, 1, 10},

		// Multi-batch tests (process multiple examples at once)
		{"Batch4-FP32-CPU", "all-MiniLM-L6-v2/onnx/model.onnx", false, 4, 1},
		{"Batch4-FP32-CoreML", "all-MiniLM-L6-v2/onnx/model.onnx", true, 4, 1},
		{"Batch4-INT8-CPU", "all-MiniLM-L6-v2/onnx/model_int8.onnx", false, 4, 1},
		{"Batch4-INT8-CoreML", "all-MiniLM-L6-v2/onnx/model_int8.onnx", true, 4, 1},

		// Heavy load tests - multiple examples with longer sequences
		{"Heavy-FP32-CPU", "all-MiniLM-L6-v2/onnx/model.onnx", false, 4, 5},
		{"Heavy-FP32-CoreML", "all-MiniLM-L6-v2/onnx/model.onnx", true, 4, 5},
		{"Heavy-INT8-CPU", "all-MiniLM-L6-v2/onnx/model_int8.onnx", false, 4, 5},
		{"Heavy-INT8-CoreML", "all-MiniLM-L6-v2/onnx/model_int8.onnx", true, 4, 5},
	}

	// Run each test case
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create session options
			sessionOptions, err := NewSessionOptions()
			if err != nil {
				t.Fatalf("Error creating session options: %v", err)
			}
			defer sessionOptions.Destroy()

			// Configure with CoreML if specified
			if tc.useCoreML {
				coreMLOptions := NewCoreMLProviderOptions()
				coreMLOptions.SetModelFormat("MLProgram")
				coreMLOptions.SetMLComputeUnits("CPUOnly")
				coreMLOptions.SetEnableOnSubgraphs(true)
				coreMLOptions.SetModelCacheDirectory("/tmp")
				err = sessionOptions.AppendExecutionProviderCoreMLV2(coreMLOptions)
				if err != nil {
					t.Fatalf("Error configuring CoreML provider: %v", err)
				}
			}

			// Create longer sequences by concatenating examples
			// For this test, we'll create one mega-example with tc.sequenceMultiplier times more tokens

			// Start with a smaller number of examples to create our dataset
			startIdx := 0
			totalTokens := 0

			// Create batches of input
			batchInputIDs := make([][]int64, tc.batchSize)
			batchAttentionMask := make([][]int64, tc.batchSize)
			batchTokenTypeIDs := make([][]int64, tc.batchSize)
			batchLengths := make([]int, tc.batchSize)

			for b := 0; b < tc.batchSize; b++ {
				// For each batch item, concatenate multiple examples
				for i := 0; i < tc.sequenceMultiplier; i++ {
					exampleIdx := (startIdx + i + b*tc.sequenceMultiplier) % numExamples

					// Add this example's tokens to the long sequence
					batchInputIDs[b] = append(batchInputIDs[b], examples[exampleIdx].InputIDs...)
					batchAttentionMask[b] = append(batchAttentionMask[b], examples[exampleIdx].AttentionMask...)
					batchTokenTypeIDs[b] = append(batchTokenTypeIDs[b], examples[exampleIdx].TokenTypeIDs...)
				}

				batchLengths[b] = len(batchInputIDs[b])
				totalTokens += batchLengths[b]
			}

			// Find the max sequence length across batches
			maxSeqLen := 0
			for _, length := range batchLengths {
				if length > maxSeqLen {
					maxSeqLen = length
				}
			}

			// Log what we've created
			t.Logf("Created %d batches with max sequence length: %d", tc.batchSize, maxSeqLen)
			t.Logf("Total tokens across all batches: %d", totalTokens)
			for b := 0; b < tc.batchSize; b++ {
				t.Logf("  Batch %d: %d tokens", b, batchLengths[b])
			}

			// Create padded tensors for the batch
			paddedInputIDs := make([]int64, tc.batchSize*maxSeqLen)
			paddedAttentionMask := make([]int64, tc.batchSize*maxSeqLen)
			paddedTokenTypeIDs := make([]int64, tc.batchSize*maxSeqLen)

			// Copy data into padded tensors
			for b := 0; b < tc.batchSize; b++ {
				batchOffset := b * maxSeqLen
				copy(paddedInputIDs[batchOffset:], batchInputIDs[b])
				copy(paddedAttentionMask[batchOffset:], batchAttentionMask[b])
				copy(paddedTokenTypeIDs[batchOffset:], batchTokenTypeIDs[b])

				// Fill attention mask appropriately (1 for tokens, 0 for padding)
				for i := 0; i < maxSeqLen; i++ {
					if i < batchLengths[b] {
						paddedAttentionMask[batchOffset+i] = 1
					} else {
						paddedAttentionMask[batchOffset+i] = 0
						paddedInputIDs[batchOffset+i] = 0     // Use padding token
						paddedTokenTypeIDs[batchOffset+i] = 0 // Use padding type
					}
				}
			}

			// Create input tensors
			inputIdsTensor, err := NewTensor(NewShape(int64(tc.batchSize), int64(maxSeqLen)), paddedInputIDs)
			if err != nil {
				t.Fatalf("Error creating input_ids tensor: %v", err)
			}
			defer inputIdsTensor.Destroy()

			attentionMaskTensor, err := NewTensor(NewShape(int64(tc.batchSize), int64(maxSeqLen)), paddedAttentionMask)
			if err != nil {
				t.Fatalf("Error creating attention_mask tensor: %v", err)
			}
			defer attentionMaskTensor.Destroy()

			tokenTypeIdsTensor, err := NewTensor(NewShape(int64(tc.batchSize), int64(maxSeqLen)), paddedTokenTypeIDs)
			if err != nil {
				t.Fatalf("Error creating token_type_ids tensor: %v", err)
			}
			defer tokenTypeIdsTensor.Destroy()

			// Create output tensor (assuming 384-dim embeddings per token)
			outputShape := NewShape(int64(tc.batchSize), int64(maxSeqLen), 384)
			outputData := make([]float32, tc.batchSize*maxSeqLen*384)
			outputTensor, err := NewTensor(outputShape, outputData)
			if err != nil {
				t.Fatalf("Error creating output tensor: %v", err)
			}
			defer outputTensor.Destroy()

			// Create session with model
			startTime := time.Now()
			session, err := NewAdvancedSession(
				tc.modelPath,
				[]string{"input_ids", "attention_mask", "token_type_ids"},
				[]string{"last_hidden_state"},
				[]Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor},
				[]Value{outputTensor},
				sessionOptions,
			)
			if err != nil {
				t.Fatalf("Error creating session: %v", err)
			}
			defer session.Destroy()

			// Report model load time
			loadTime := time.Since(startTime)
			t.Logf("Model load time: %.4f seconds", loadTime.Seconds())

			// First run (with compilation)
			startTime = time.Now()
			err = session.Run()
			if err != nil {
				t.Fatalf("Error on first inference: %v", err)
			}
			firstRunTime := time.Since(startTime)
			t.Logf("First run time: %.4f ms", float64(firstRunTime.Microseconds())/1000.0)

			// Now run multiple times for accurate timing
			const numRuns = 5
			totalInferenceTime := time.Duration(0)

			for i := 0; i < numRuns; i++ {
				startTime = time.Now()
				err = session.Run()
				if err != nil {
					t.Fatalf("Error on run %d: %v", i+1, err)
				}
				runTime := time.Since(startTime)
				totalInferenceTime += runTime

				// Log each run's performance
				t.Logf("Run %d: %.4f ms", i+1, float64(runTime.Microseconds())/1000.0)
			}

			// Calculate average time and throughput
			avgTime := totalInferenceTime / time.Duration(numRuns)
			t.Logf("Average inference time (after first run): %.4f ms", float64(avgTime.Microseconds())/1000.0)

			// Calculate tokens per second based on real token count (not including padding)
			tokensPerSecond := float64(totalTokens*numRuns) / totalInferenceTime.Seconds()
			t.Logf("Average tokens/second: %.2f", tokensPerSecond)

			// Memory usage
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			t.Logf("Memory usage: %d MB", m.Alloc/1024/1024)
		})
	}
}

// TestComplexExamplesBenchmark benchmarks performance with the more challenging examples
// created by our modified Python script
func TestComplexExamplesBenchmark(t *testing.T) {
	initializeRuntimeWithVerboseLogs(t)
	defer cleanupRuntime(t)

	// Read the complex tokenized data from JSON
	jsonData, err := os.ReadFile("tokenized_complex_data.json")
	if err != nil {
		t.Fatalf("Error reading complex tokenized data: %v", err)
	}

	// Parse complex JSON data
	var complexData ComplexData
	if err := json.Unmarshal(jsonData, &complexData); err != nil {
		t.Fatalf("Error parsing complex JSON: %v", err)
	}

	// Validate data
	if len(complexData.Regular) == 0 && len(complexData.Long) == 0 &&
		len(complexData.MaxLength) == 0 && len(complexData.Batched) == 0 {
		t.Fatalf("No examples found in complex tokenized data")
	}

	// Log stats about loaded data
	t.Logf("Loaded %d regular examples", len(complexData.Regular))
	t.Logf("Loaded %d long examples", len(complexData.Long))
	t.Logf("Loaded %d max-length examples", len(complexData.MaxLength))
	t.Logf("Loaded %d batched examples", len(complexData.Batched))

	// Define test configurations
	testConfigs := []struct {
		name         string
		modelPath    string
		useCoreML    bool
		computeUnits string
	}{
		{"INT8-CPU", "all-MiniLM-L6-v2/onnx/model_int8.onnx", false, ""},
		{"INT8-CoreML-CPUOnly", "all-MiniLM-L6-v2/onnx/model_int8.onnx", true, "CPUOnly"},
		{"INT8-CoreML-CPUAndGPU", "all-MiniLM-L6-v2/onnx/model_int8.onnx", true, "CPUAndGPU"},
		{"INT8-CoreML-ALL", "all-MiniLM-L6-v2/onnx/model_int8.onnx", true, "ALL"},
	}

	// Run benchmark for each test configuration
	for _, config := range testConfigs {
		// Test with long examples
		t.Run(config.name+"-LongExamples", func(t *testing.T) {
			benchmarkLongExamples(t, complexData.Long, config.modelPath, config.useCoreML, config.computeUnits)
		})

		// Test with max length examples
		t.Run(config.name+"-MaxLengthExamples", func(t *testing.T) {
			benchmarkLongExamples(t, complexData.MaxLength, config.modelPath, config.useCoreML, config.computeUnits)
		})

		// Test with batched examples
		t.Run(config.name+"-BatchedExamples", func(t *testing.T) {
			benchmarkBatchedExamples(t, complexData.Batched, config.modelPath, config.useCoreML, config.computeUnits)
		})
	}
}

// benchmarkLongExamples handles benchmarking for single examples (long or max-length)
func benchmarkLongExamples(t *testing.T, examples []Example, modelPath string, useCoreML bool, computeUnits string) {
	if len(examples) == 0 {
		t.Skip("No long examples to benchmark")
	}

	// Select a subset of examples to test (to keep runtime reasonable)
	numExamplesToTest := 5
	if numExamplesToTest > len(examples) {
		numExamplesToTest = len(examples)
	}

	// Create session options
	sessionOptions, err := NewSessionOptions()
	if err != nil {
		t.Fatalf("Error creating session options: %v", err)
	}
	defer sessionOptions.Destroy()

	// Configure with CoreML if specified
	if useCoreML {
		coreMLOptions := NewCoreMLProviderOptions()
		coreMLOptions.SetModelFormat(CoreMLModelFormatMLProgram)
		coreMLOptions.SetMLComputeUnits(CoreMLComputeUnits(computeUnits))
		coreMLOptions.SetEnableOnSubgraphs(true)
		coreMLOptions.SetModelCacheDirectory("/tmp")
		err = sessionOptions.AppendExecutionProviderCoreMLV2(coreMLOptions)
		if err != nil {
			t.Fatalf("Error configuring CoreML provider: %v", err)
		}
	}

	// Use a random subset of examples
	rand.Seed(time.Now().UnixNano())
	indices := rand.Perm(len(examples))[:numExamplesToTest]

	// Track total tokens and performance
	totalTokens := 0
	totalInferenceTime := time.Duration(0)
	var firstRunTime time.Duration

	// Run inference on selected examples
	for i, exampleIdx := range indices {
		example := examples[exampleIdx]
		seqLen := len(example.InputIDs)
		totalTokens += seqLen

		// Create input tensors
		inputIdsTensor, err := NewTensor(NewShape(1, int64(seqLen)), example.InputIDs)
		if err != nil {
			t.Fatalf("Error creating input_ids tensor: %v", err)
		}
		defer inputIdsTensor.Destroy()

		attentionMaskTensor, err := NewTensor(NewShape(1, int64(seqLen)), example.AttentionMask)
		if err != nil {
			t.Fatalf("Error creating attention_mask tensor: %v", err)
		}
		defer attentionMaskTensor.Destroy()

		tokenTypeIdsTensor, err := NewTensor(NewShape(1, int64(seqLen)), example.TokenTypeIDs)
		if err != nil {
			t.Fatalf("Error creating token_type_ids tensor: %v", err)
		}
		defer tokenTypeIdsTensor.Destroy()

		// Create output tensor (384-dim embeddings)
		outputShape := NewShape(1, int64(seqLen), 384)
		outputData := make([]float32, seqLen*384)
		outputTensor, err := NewTensor(outputShape, outputData)
		if err != nil {
			t.Fatalf("Error creating output tensor: %v", err)
		}
		defer outputTensor.Destroy()

		// Create session
		startTime := time.Now()
		session, err := NewAdvancedSession(
			modelPath,
			[]string{"input_ids", "attention_mask", "token_type_ids"},
			[]string{"last_hidden_state"},
			[]Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor},
			[]Value{outputTensor},
			sessionOptions,
		)
		if err != nil {
			t.Fatalf("Error creating session: %v", err)
		}
		defer session.Destroy()

		// Log model load time for first example
		if i == 0 {
			loadTime := time.Since(startTime)
			t.Logf("Model load time: %.4f seconds", loadTime.Seconds())
		}

		// Run inference
		startInference := time.Now()
		err = session.Run()
		if err != nil {
			t.Fatalf("Error running inference: %v", err)
		}
		inferenceTime := time.Since(startInference)

		// Track performance
		if i == 0 {
			firstRunTime = inferenceTime
			t.Logf("First run (compilation) time: %.4f ms", float64(inferenceTime.Microseconds())/1000.0)
		} else {
			totalInferenceTime += inferenceTime
			tokensPerSecond := float64(seqLen) / inferenceTime.Seconds()
			t.Logf("Example %d: %d tokens in %.4f ms (%.2f tokens/sec)",
				exampleIdx, seqLen, float64(inferenceTime.Microseconds())/1000.0, tokensPerSecond)
		}
	}

	// Calculate overall stats
	if numExamplesToTest > 1 {
		avgTime := totalInferenceTime / time.Duration(numExamplesToTest-1) // exclude first run
		t.Logf("Average inference time (after first run): %.4f ms", float64(avgTime.Microseconds())/1000.0)
		t.Logf("First run (compilation) time: %.4f ms", float64(firstRunTime.Microseconds())/1000.0)

		tokensPerSecond := float64(totalTokens) / totalInferenceTime.Seconds()
		t.Logf("Average tokens/second: %.2f", tokensPerSecond)
	}
}

// benchmarkBatchedExamples handles benchmarking for batched examples
func benchmarkBatchedExamples(t *testing.T, batchExamples []BatchExample, modelPath string, useCoreML bool, computeUnits string) {
	if len(batchExamples) == 0 {
		t.Skip("No batched examples to benchmark")
	}

	// Select a subset of examples to test (to keep runtime reasonable)
	numBatchesToTest := 5
	if numBatchesToTest > len(batchExamples) {
		numBatchesToTest = len(batchExamples)
	}

	// Create session options
	sessionOptions, err := NewSessionOptions()
	if err != nil {
		t.Fatalf("Error creating session options: %v", err)
	}
	defer sessionOptions.Destroy()

	// Configure with CoreML if specified
	if useCoreML {
		coreMLOptions := NewCoreMLProviderOptions()
		coreMLOptions.SetModelFormat(CoreMLModelFormatMLProgram)
		coreMLOptions.SetMLComputeUnits(CoreMLComputeUnits(computeUnits))
		coreMLOptions.SetEnableOnSubgraphs(true)
		coreMLOptions.SetModelCacheDirectory("/tmp")
		err = sessionOptions.AppendExecutionProviderCoreMLV2(coreMLOptions)
		if err != nil {
			t.Fatalf("Error configuring CoreML provider: %v", err)
		}
	}

	// Use a random subset of examples
	rand.Seed(time.Now().UnixNano())
	indices := rand.Perm(len(batchExamples))[:numBatchesToTest]

	// Track total tokens and performance
	totalTokens := 0
	totalInferenceTime := time.Duration(0)
	var firstRunTime time.Duration

	// Run inference on selected batch examples
	for i, batchIdx := range indices {
		batchExample := batchExamples[batchIdx]
		batchSize := batchExample.BatchSize

		// Find max sequence length in this batch
		maxSeqLen := 0
		for _, seq := range batchExample.InputIDs {
			if len(seq) > maxSeqLen {
				maxSeqLen = len(seq)
			}
		}

		// Count total actual tokens (not including padding)
		batchTokens := 0
		for _, seq := range batchExample.InputIDs {
			batchTokens += len(seq)
		}
		totalTokens += batchTokens

		// Create padded input tensors
		paddedInputIDs := make([]int64, batchSize*maxSeqLen)
		paddedAttentionMask := make([]int64, batchSize*maxSeqLen)
		paddedTokenTypeIDs := make([]int64, batchSize*maxSeqLen)

		// Copy data and add padding
		for b := 0; b < batchSize; b++ {
			offset := b * maxSeqLen
			seqLen := len(batchExample.InputIDs[b])

			// Copy actual data
			copy(paddedInputIDs[offset:], batchExample.InputIDs[b])
			copy(paddedAttentionMask[offset:], batchExample.AttentionMask[b])
			copy(paddedTokenTypeIDs[offset:], batchExample.TokenTypeIDs[b])

			// Add padding
			for j := seqLen; j < maxSeqLen; j++ {
				paddedInputIDs[offset+j] = 0      // Padding token
				paddedAttentionMask[offset+j] = 0 // Padding mask
				paddedTokenTypeIDs[offset+j] = 0  // Padding type
			}
		}

		// Create tensors
		inputIdsTensor, err := NewTensor(NewShape(int64(batchSize), int64(maxSeqLen)), paddedInputIDs)
		if err != nil {
			t.Fatalf("Error creating input_ids tensor: %v", err)
		}
		defer inputIdsTensor.Destroy()

		attentionMaskTensor, err := NewTensor(NewShape(int64(batchSize), int64(maxSeqLen)), paddedAttentionMask)
		if err != nil {
			t.Fatalf("Error creating attention_mask tensor: %v", err)
		}
		defer attentionMaskTensor.Destroy()

		tokenTypeIdsTensor, err := NewTensor(NewShape(int64(batchSize), int64(maxSeqLen)), paddedTokenTypeIDs)
		if err != nil {
			t.Fatalf("Error creating token_type_ids tensor: %v", err)
		}
		defer tokenTypeIdsTensor.Destroy()

		// Create output tensor (384-dim embeddings)
		outputShape := NewShape(int64(batchSize), int64(maxSeqLen), 384)
		outputData := make([]float32, batchSize*maxSeqLen*384)
		outputTensor, err := NewTensor(outputShape, outputData)
		if err != nil {
			t.Fatalf("Error creating output tensor: %v", err)
		}
		defer outputTensor.Destroy()

		// Create session
		startTime := time.Now()
		session, err := NewAdvancedSession(
			modelPath,
			[]string{"input_ids", "attention_mask", "token_type_ids"},
			[]string{"last_hidden_state"},
			[]Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor},
			[]Value{outputTensor},
			sessionOptions,
		)
		if err != nil {
			t.Fatalf("Error creating session: %v", err)
		}
		defer session.Destroy()

		// Log model load time for first example
		if i == 0 {
			loadTime := time.Since(startTime)
			t.Logf("Model load time: %.4f seconds", loadTime.Seconds())
		}

		// Run inference
		startInference := time.Now()
		err = session.Run()
		if err != nil {
			t.Fatalf("Error running inference: %v", err)
		}
		inferenceTime := time.Since(startInference)

		// Track performance
		if i == 0 {
			firstRunTime = inferenceTime
			t.Logf("First run (compilation) time: %.4f ms", float64(inferenceTime.Microseconds())/1000.0)
		} else {
			totalInferenceTime += inferenceTime
			tokensPerSecond := float64(batchTokens) / inferenceTime.Seconds()
			t.Logf("Batch %d: batch_size=%d, tokens=%d, time=%.4f ms (%.2f tokens/sec)",
				batchIdx, batchSize, batchTokens, float64(inferenceTime.Microseconds())/1000.0, tokensPerSecond)
		}
	}

	// Calculate overall stats
	if numBatchesToTest > 1 {
		avgTime := totalInferenceTime / time.Duration(numBatchesToTest-1) // exclude first run
		t.Logf("Average inference time (after first run): %.4f ms", float64(avgTime.Microseconds())/1000.0)
		t.Logf("First run (compilation) time: %.4f ms", float64(firstRunTime.Microseconds())/1000.0)

		tokensPerSecond := float64(totalTokens) / totalInferenceTime.Seconds()
		t.Logf("Average tokens/second: %.2f", tokensPerSecond)
	}
}
