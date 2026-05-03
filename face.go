package facial

import "image"

type Face struct {
	Rect    image.Rectangle
	Shape   Shape
	Score   float32
	Feature []float32
}

func (f *Face) Square(margin float64, srcWidth, srcHeight int) image.Rectangle {
	dx := f.Rect.Dx()
	dy := f.Rect.Dy()
	var x0, y0, x1, y1 int
	if dx < dy {
		offsetY := int(float64(dy) * margin)
		y0 = f.Rect.Min.Y - offsetY
		y1 = f.Rect.Max.Y + offsetY
		length := offsetY<<1 + dy
		x0 = f.Rect.Min.X - (length-dx)>>1
		x1 = x0 + length
	} else {
		offsetX := int(float64(dx) * margin)
		x0 = f.Rect.Min.X - offsetX
		x1 = f.Rect.Max.X + offsetX
		length := offsetX<<1 + dx
		y0 = f.Rect.Min.Y - (length-dy)>>1
		y1 = y0 + length
	}
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > srcWidth {
		x1 = srcWidth
	}
	if y1 > srcHeight {
		y1 = srcHeight
	}
	return image.Rect(x0, y0, x1, y1)
}
