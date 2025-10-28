package main

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
)

func activationFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

type NeuralNetwork struct {
	Layers    []matrix
	LearnRate float64
	Weights   []matrix
}

func (nn *NeuralNetwork) Save(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()

    encoder := gob.NewEncoder(file)
    err = encoder.Encode(nn.Weights)
    if err != nil {
        return err
    }
    err = encoder.Encode(nn.LearnRate)
    if err != nil {
        return err
    }
    return nil
}

func (nn *NeuralNetwork) Load(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()

    decoder := gob.NewDecoder(file)
    err = decoder.Decode(&nn.Weights)
    if err != nil {
        return err
    }
    err = decoder.Decode(&nn.LearnRate)
    if err != nil {
        return err
    }
    
    nn.Layers = make([]matrix, len(nn.Weights)+1)
    return nil
}

func (nn *NeuralNetwork) init(nodesInLayers []int, learnRate float64) {
	if len(nodesInLayers) < 3 {
		panic("atleast 3 layers are expected")
	}

	nn.LearnRate = learnRate
	weights := make([]matrix, len(nodesInLayers)-1)

	for i := 0; i < len(nodesInLayers)-1; i++ {
		row := nodesInLayers[i+1]
		col := nodesInLayers[i]
		stdDeviation := math.Pow(float64(col), -0.5)

		matrix := make([][]float64, row)

		for r := range row {
			rowArr := make([]float64, col)
			matrix[r] = rowArr

			for c := range col {
				rn := rand.Float64()*2 - 1 // range [-1, 1)
				rn *= stdDeviation
				rowArr[c] = rn
			}
		}
		weights[i] = matrix
	}

	layers := make([]matrix, len(nodesInLayers))
	nn.Layers = layers
	nn.Weights = weights
}

func (nn *NeuralNetwork) train(query []float64, result []float64) {
	input := Transpose([][]float64{query})
	target := Transpose([][]float64{result})

	nn.Layers[0] = input
	for i := range nn.Weights {
		result := MulMat(nn.Weights[i], nn.Layers[i])
		nn.Layers[i+1] = result.transform(func(x float64, i, j int) float64 {
			return activationFunc(x)
		})
	}

	layer_err := target.transform(func(x float64, i, j int) float64 {
		return x - nn.Layers[len(nn.Layers)-1][i][j]
	})

	for i := len(nn.Layers) - 1; i > 0; i-- {
		layer := nn.Layers[i]
		prevLayer := nn.Layers[i-1]

		actDeriv := layer.transform(func(x float64, r, c int) float64 {
			return x * (1.0 - x)
		})

		delta := layer_err.Mul(actDeriv)

		weights_grads := MulMat(delta, Transpose(prevLayer))

		// error in prev layer
		weights_T := Transpose(nn.Weights[i-1])
		layer_err = MulMat(weights_T, delta)

		nn.Weights[i-1] = weights_grads.transform(func(x float64, k, j int) float64 {
			return nn.Weights[i-1][k][j] + (x * nn.LearnRate)
		})
	}
}

func (nn *NeuralNetwork) query(query []float64) []float64 {
	input := Transpose([][]float64{query})
	for i := range len(nn.Layers) - 1 {
		result := MulMat(nn.Weights[i], input)
		input = result.transform(func(x float64, _, _ int) float64 {
			return activationFunc(x)
		})
	}
	return Transpose(input)[0]
}

