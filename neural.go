package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
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

		actDeriv := layer.transform(func(x float64, _, _ int) float64 {
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

type HandWrittenNum struct {
	number  int
	bytemap [784]uint8
}

func getData_HandWrittenNum(datafile string) []HandWrittenNum  {	
	data := make([]HandWrittenNum, 0, 100)
	
	file, err := os.Open(datafile)
	if err != nil {
		log.Fatal(err)
	}

	reader := bufio.NewReader(file)
	for {
		line, _, err := reader.ReadLine()
	
		if len(line) > 0 {
			h := HandWrittenNum{}
			h.number = int(line[0]) - 48
	
			vals := string(line[2:])
			valsArr := strings.Split(vals, ",")
			for i, v := range valsArr {
				n, _ := strconv.ParseUint(v, 10, 8)
				h.bytemap[i] = uint8(n)
			}
	
			data = append(data, h)
		}
	
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
	}
	return data
}

func main() {
	var nn NeuralNetwork
	nn.init([]int{784, 100, 10}, 0.3)

	data := getData_HandWrittenNum("mnist_train_100.csv")

	first := data[0]
	q := make([]float64, 784)
	for i := range q {
		v := first.bytemap[i]
		q[i] = (float64(v)/255)*0.99 + 0.01
	}
	
	fmt.Println(nn.query(q), first.number)
	
	t := make([]float64, 10)
	
	for _, datum := range data {
		for j := range q {
			v := datum.bytemap[j]
			q[j] = (float64(v)/255)*0.99 + 0.01
		}
		for i := range t {
			t[i] = 0.01
		}
		ans := datum.number
		t[ans] = 0.99

		nn.train(q, t)
	}

	q = make([]float64, 784)
	for i := range q {
		v := first.bytemap[i]
		q[i] = (float64(v)/255)*0.99 + 0.01
	}

	ans := nn.query(q)
	for i, a := range ans {
		fmt.Println(i, a)
	}
}

