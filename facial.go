package facial

import (
	"errors"
	"image"
	"math"

	"github.com/oxzjh/cv"
)

type IDetector interface {
	Detect(*cv.Mat) ([]*cv.Box, error)
	Close()
}

type IRecognizer interface {
	Extract(*cv.Mat) ([]float32, error)
	Close()
}

type Facial struct {
	detector   IDetector
	recognizer IRecognizer
}

func (f *Facial) Detect(m *cv.Mat) ([]*Face, error) {
	if m.Cols() < 32 || m.Rows() < 32 {
		return nil, errors.New("read image failed")
	}
	boxes, err := f.detector.Detect(m)
	if err != nil {
		return nil, err
	}
	faces := make([]*Face, len(boxes))
	for i, box := range boxes {
		landmarks := box.Data.(*[10]float32)
		aligned := m.AlignCropFace(*landmarks)
		feature, err := f.recognizer.Extract(aligned)
		aligned.Free()
		if err != nil {
			return nil, err
		}
		shape := make(Shape, 5)
		for j := 0; j < 5; j++ {
			offset := j << 1
			shape[j].X = int(landmarks[offset] + 0.5)
			shape[j].Y = int(landmarks[offset+1] + 0.5)
		}
		faces[i] = &Face{image.Rect(
			int(box.X0+0.5),
			int(box.Y0+0.5),
			int(box.X1+0.5),
			int(box.Y1+0.5),
		), shape, box.Score, feature}
	}
	return faces, nil
}

func (f *Facial) DetectFile(file string) ([]*Face, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	defer m.Free()
	return f.Detect(m)
}

func (f *Facial) DetectImage(image []byte) ([]*Face, error) {
	m := cv.IMDecode(image, cv.FLAG_COLOR)
	defer m.Free()
	return f.Detect(m)
}

func (f *Facial) Close() {
	f.detector.Close()
	f.recognizer.Close()
}

func New(detector IDetector, recognizer IRecognizer) *Facial {
	return &Facial{detector, recognizer}
}

func GetBoxes(rows, cols, n int, width, height, scaleX, scaleY, scoreThreshold, iouThreshold float32, scores, scale0, offset0, shapes []float32) (boxes []*cv.Box) {
	scale1 := scale0[n:]
	offset1 := offset0[n:]
	for y := 0; y < rows; y++ {
		offset := y * cols
		for x := 0; x < cols; x++ {
			index := offset + x
			if score := scores[index]; score > scoreThreshold {
				s0 := float32(math.Exp(float64(scale0[index]))) * 4
				s1 := float32(math.Exp(float64(scale1[index]))) * 4
				o0 := (float32(y) + offset0[index] + 0.5) * 4
				o1 := (float32(x) + offset1[index] + 0.5) * 4
				var data [10]float32
				box := &cv.Box{
					ID:    index,
					X0:    cv.Clipf(o1-s1*0.5, 0, width),
					Y0:    cv.Clipf(o0-s0*0.5, 0, height),
					Score: score,
					Data:  &data,
				}
				box.X1 = cv.Minf(box.X0+s1, width)
				box.Y1 = cv.Minf(box.Y0+s0, height)
				for z := 0; z < 10; z += 2 {
					sOffset := z*n + index
					data[z] = box.X0 + shapes[sOffset+n]*s1
					data[z+1] = box.Y0 + shapes[sOffset]*s0
				}
				boxes = append(boxes, box)
			}
		}
		boxes = cv.NMS(boxes, iouThreshold)
	}
	if len(boxes) > 0 {
		for _, box := range boxes {
			box.X0 *= scaleX
			box.Y0 *= scaleY
			box.X1 *= scaleX
			box.Y1 *= scaleY
			data := box.Data.(*[10]float32)
			for i := 0; i < 10; i += 2 {
				data[i] *= scaleX
				data[i+1] *= scaleY
			}
		}
	}
	return
}
