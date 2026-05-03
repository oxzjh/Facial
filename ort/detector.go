package ort

import (
	"github.com/oxzjh/cv"
	"github.com/oxzjh/facial"
	"github.com/oxzjh/ort"
)

type Detector struct {
	engine *ort.Ort
	opts   *options
}

func (d *Detector) Detect(m *cv.Mat) ([]*cv.Box, error) {
	resized, ratio := m.Resize32(d.opts.maxSize, cv.INTER_CUBIC)
	inputBlob := resized.BlobFromImage(1, cv.Size{}, cv.Scalar{}, true, false)
	resized.Free()
	defer inputBlob.Free()
	result, err := d.engine.InferWithShape(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), []int64{1, 3, int64(ratio.Height), int64(ratio.Width)}, ort.TypeFloat)
	if err != nil {
		return nil, err
	}
	defer result.Free()
	rows := ratio.Height >> 2
	cols := ratio.Width >> 2
	n := rows * cols
	scores, err := result.GetFloat32(0, n, false)
	if err != nil {
		return nil, err
	}
	scale0, err := result.GetFloat32(1, n<<1, false)
	if err != nil {
		return nil, err
	}
	offset0, err := result.GetFloat32(2, n<<1, false)
	if err != nil {
		return nil, err
	}
	shapes, err := result.GetFloat32(3, n*10, false)
	if err != nil {
		return nil, err
	}
	return facial.GetBoxes(rows, cols, n, float32(ratio.Width), float32(ratio.Height), ratio.ScaleX, ratio.ScaleY, d.opts.score, d.opts.iou, scores, scale0, offset0, shapes), nil
}

func (d *Detector) Close() {
	if d.engine != nil {
		d.engine.Close()
		d.engine = nil
	}
}

func NewDetector(model string, threads int, opts ...Option) (facial.IDetector, error) {
	o, err := initOptions(opts)
	if err != nil {
		return nil, err
	}
	engine, err := ort.New(model, "facial_detector", threads)
	if err != nil {
		return nil, err
	}
	return &Detector{engine, o}, nil
}

func NewDetectorWithBuffer(model []byte, threads int, opts ...Option) (facial.IDetector, error) {
	o, err := initOptions(opts)
	if err != nil {
		return nil, err
	}
	engine, err := ort.NewWithBuffer(model, "facial_detector", threads)
	if err != nil {
		return nil, err
	}
	return &Detector{engine, o}, nil
}
