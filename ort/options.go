package ort

import "errors"

type options struct {
	maxSize int
	score   float32
	iou     float32
}

type Option func(*options)

func WithMaxSize(maxSize int) Option {
	return func(o *options) {
		o.maxSize = maxSize
	}
}

func WithScore(score float32) Option {
	return func(o *options) {
		o.score = score
	}
}

func WithIou(iou float32) Option {
	return func(o *options) {
		o.iou = iou
	}
}

func initOptions(opts []Option) (*options, error) {
	o := &options{
		maxSize: 640,
		score:   0.75,
		iou:     0.3,
	}
	for _, opt := range opts {
		opt(o)
	}
	if o.maxSize%32 != 0 {
		return nil, errors.New("maxSize must be a multiple of 32")
	}
	return o, nil
}
