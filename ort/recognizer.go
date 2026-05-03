package ort

import (
	"github.com/oxzjh/cv"
	"github.com/oxzjh/facial"
	"github.com/oxzjh/ort"
	"github.com/oxzjh/vector"
)

type Recognizer struct {
	engine     *ort.Ort
	outputDims int
}

func (r *Recognizer) Extract(aligned *cv.Mat) ([]float32, error) {
	inputBlob := aligned.BlobFromImage(1/128.0, cv.Size{}, cv.NewScalar(127.5, 127.5, 127.5, 0), true, false)
	defer inputBlob.Free()
	result, err := r.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat)
	if err != nil {
		return nil, err
	}
	defer result.Free()
	feature, err := result.GetFloat32(0, r.outputDims, true)
	if err != nil {
		return nil, err
	}
	vector.NormalizeSelf(feature)
	return feature, nil
}

func (r *Recognizer) Close() {
	if r.engine != nil {
		r.engine.Close()
		r.engine = nil
	}
}

func NewRecognizer(model string, threads int) (facial.IRecognizer, error) {
	engine, err := ort.New(model, "facial_recognizer", threads)
	if err != nil {
		return nil, err
	}
	return &Recognizer{engine, int(engine.OutputDims[0][1])}, nil
}

func NewRecognizerWithBuffer(model []byte, threads int) (facial.IRecognizer, error) {
	engine, err := ort.NewWithBuffer(model, "facial_recognizer", threads)
	if err != nil {
		return nil, err
	}
	return &Recognizer{engine, int(engine.OutputDims[0][1])}, nil
}
