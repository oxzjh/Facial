package facial

import (
	"image"
	"math"
)

type Shape []image.Point

func (s Shape) LeftEyeRatio() float64 {
	return eyeRatio(s[36:42])
}

func (s Shape) RightEyeRatio() float64 {
	return eyeRatio(s[42:48])
}

func (s Shape) MouthRatio() float64 {
	return magnitude(s[62], s[66]) / magnitude(s[51], s[57])
}

func (s Shape) NoseRatio() (left, right float64) {
	m := magnitude(s[2], s[14])
	left = magnitude(s[2], s[30]) / m
	right = magnitude(s[30], s[14]) / m
	return
}

func eyeRatio(p []image.Point) float64 {
	return (magnitude(p[1], p[5]) + magnitude(p[2], p[4])) * 0.5 / magnitude(p[0], p[3])
}

func magnitude(p1, p2 image.Point) float64 {
	dx := p1.X - p2.X
	dy := p1.Y - p2.Y
	return math.Sqrt(float64(dx*dx) + float64(dy*dy))
}
