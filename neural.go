package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

func activationFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

type NeuralNetwork struct {
	Layers    []matrix
	LearnRate float64
	Weights   []matrix
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
	for i := range len(nn.Layers) - 1 {
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
		prevlayer := nn.Layers[i-1]

		layer_1 := layer.transform(func(x float64, i, j int) float64 { return 1.0 - x })
		// layer_err * layer * (1.0 - layer) . prevlayer
		weights_grads := MulMat(layer_err.Mul(layer, layer_1), Transpose(prevlayer))

		weights_T := Transpose(nn.Weights[i-1])

		nn.Weights[i-1] = weights_grads.
			transform(func(x float64, k, j int) float64 { return nn.Weights[i-1][k][j] + (2 * x * nn.LearnRate) })

		// error in prev layer
		layer_err = MulMat(weights_T, layer_err)
	}
}

func (nn *NeuralNetwork) query(query []float64) []float64 {
	input := Transpose([][]float64{query})
	for i := range len(nn.Layers) - 1 {
		result := MulMat(nn.Weights[i], input)
		for i, v := range result {
			input[i][0] = activationFunc(v[0])
		}
	}
	return Transpose(input)[0]
}

func main() {
	// var nn NeuralNetwork
	// nn.init([]int{3, 3, 3}, 1)
	// fmt.Println(nn.query([]float64{1, 2, 3}))
	// for i := range 1000 {
	// 	if i == 500 {
	// 		fmt.Println(nn.query([]float64{1, 2, 3}))
	// 	}
	// 	nn.train([]float64{1, 2, 3}, []float64{0.1, 0.2, 0.3})
	// }
	// fmt.Println(nn.query([]float64{1, 2, 3}))

	data := `7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0`
	dataArr := strings.Split(data, ",")

	for i,v := range dataArr[1:] {
		n,_ := strconv.Atoi(v)
		if i % 28 == 0 {
			fmt.Print("\n")
		}
		if n > 128 {
			fmt.Print("*")
		} else {	
			fmt.Print(" ")
		}
	}
	fmt.Print("\n")
}
