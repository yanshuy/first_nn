package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"
)

type HandWrittenNum struct {
	number  int
	bytemap [784]uint8
}

func getData_HandWrittenNum(datafile string, hasHeader bool) []HandWrittenNum {
	data := make([]HandWrittenNum, 0, 10)

	file, err := os.Open(datafile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	if hasHeader {
		reader.ReadLine()
	}
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

	modelFile := "neural_network.gob"
	err := nn.Load(modelFile)
	if err != nil {
		fmt.Println("No saved model found. Training new model...")
		nn.init([]int{784, 100, 10}, 0.3)
		data := getData_HandWrittenNum("mnist_train.csv", true)

		q := make([]float64, 784)
		t := make([]float64, 10)
		now := time.Now()
		for idx, datum := range data {
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
			if (idx+1)%1000 == 0 {
				fmt.Printf("Trained %d examples in %.2fs\n", idx+1, time.Since(now).Seconds())
				now = time.Now()
			}
		}

		err = nn.Save(modelFile)
		if err != nil {
			log.Fatal("Error saving model:", err)
		}
	} else {
		fmt.Println("Loaded existing model from", modelFile)
	}

	dataTest := getData_HandWrittenNum("mnist_train.csv", false)

	correct := 0
	total := 0
	for _, datum := range dataTest {
		q := make([]float64, 784)
		// fmt.Println("image for ans", datum.number)
		for i := range q {
			v := datum.bytemap[i]
			q[i] = (float64(v)/255)*0.99 + 0.01

			// if i % 28 == 0 {
			// 	fmt.Print("\n")
			// }
			// if v > 200 {
			// 	fmt.Print("@")
			// } else if v > 120 {
			// 	fmt.Print("c")
			// } else {
			// 	fmt.Print(" ")
			// }
		}
		// fmt.Print("\n")

		ans := nn.query(q)

		max := slices.Max(ans)
		num := slices.Index(ans, max)
		if num == datum.number {
			correct++
		}
		total++
		// for i, a := range ans {
		// 	fmt.Printf("%d %.2f\n", i, a)
		// }
		// fmt.Print("\n")
	}
	fmt.Println(correct, "out of", total)
}
