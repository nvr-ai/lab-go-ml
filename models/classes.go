// Package models - provides the classes and helper utilities for the model class mapping.
package models

import (
	"fmt"

	"github.com/nvr-ai/go-ml/models/model"
)

// ClassName is the name of an output class.
type ClassName string

// Class represents a mapping between a class name and it's
// relationship identity to model families.
type Class struct {
	// The family of the model that the class belongs to.
	Family model.Family
	// The name of the class (e.g. "person", "car", "truck", "bus", "motorcycle", "bicycle")
	Name ClassName
	// The index of the class (e.g. 0 for "person", 1 for "car", 2 for "truck", 3 for "bus", 4 for
	// "motorcycle", 5 for "bicycle")
	Index int
}

// Classes provides a comprehensive mapping between class names and their indices across different
// model families. This mapping enables seamless translation between different model formats and
// their respective class indexing schemes.
var Classes = map[ClassName]map[model.Family]Class{
	"__background__": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "__background__", Index: 0},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "__background__", Index: 0},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "__background__", Index: 0},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "__background__", Index: 0},
	},
	"person": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "person", Index: 1},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "person", Index: 0},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "person", Index: 1},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "person", Index: 15},
	},
	"bicycle": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bicycle", Index: 2},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bicycle", Index: 1},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bicycle", Index: 2},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "bicycle", Index: 2},
	},
	"car": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "car", Index: 3},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "car", Index: 2},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "car", Index: 3},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "car", Index: 7},
	},
	"motorcycle": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "motorcycle", Index: 4},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "motorcycle", Index: 3},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "motorcycle", Index: 4},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "motorbike", Index: 14},
	},
	"motorbike": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "motorcycle", Index: 4},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "motorcycle", Index: 3},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "motorcycle", Index: 4},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "motorbike", Index: 14},
	},
	"airplane": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "airplane", Index: 5},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "airplane", Index: 4},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "airplane", Index: 5},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "aeroplane", Index: 1},
	},
	"aeroplane": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "airplane", Index: 5},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "airplane", Index: 4},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "airplane", Index: 5},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "aeroplane", Index: 1},
	},
	"bus": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bus", Index: 6},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bus", Index: 5},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bus", Index: 6},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "bus", Index: 6},
	},
	"train": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "train", Index: 7},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "train", Index: 6},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "train", Index: 7},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "train", Index: 19},
	},
	"truck": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "truck", Index: 8},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "truck", Index: 7},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "truck", Index: 8},
	},
	"boat": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "boat", Index: 9},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "boat", Index: 8},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "boat", Index: 9},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "boat", Index: 4},
	},
	"traffic light": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "traffic light", Index: 10},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "traffic light", Index: 9},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "traffic light", Index: 10},
	},
	"fire hydrant": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "fire hydrant", Index: 11},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "fire hydrant", Index: 10},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "fire hydrant", Index: 11},
	},
	"stop sign": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "stop sign", Index: 12},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "stop sign", Index: 11},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "stop sign", Index: 12},
	},
	"parking meter": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "parking meter", Index: 13},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "parking meter", Index: 12},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "parking meter", Index: 13},
	},
	"bench": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bench", Index: 14},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bench", Index: 13},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bench", Index: 14},
	},
	"bird": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bird", Index: 15},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bird", Index: 14},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bird", Index: 15},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "bird", Index: 3},
	},
	"cat": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "cat", Index: 16},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "cat", Index: 15},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "cat", Index: 16},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "cat", Index: 8},
	},
	"dog": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "dog", Index: 17},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "dog", Index: 16},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "dog", Index: 17},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "dog", Index: 12},
	},
	"horse": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "horse", Index: 18},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "horse", Index: 17},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "horse", Index: 18},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "horse", Index: 13},
	},
	"sheep": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "sheep", Index: 19},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "sheep", Index: 18},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "sheep", Index: 19},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "sheep", Index: 17},
	},
	"cow": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "cow", Index: 20},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "cow", Index: 19},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "cow", Index: 20},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "cow", Index: 10},
	},
	"elephant": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "elephant", Index: 21},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "elephant", Index: 20},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "elephant", Index: 21},
	},
	"bear": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bear", Index: 22},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bear", Index: 21},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bear", Index: 22},
	},
	"zebra": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "zebra", Index: 23},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "zebra", Index: 22},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "zebra", Index: 23},
	},
	"giraffe": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "giraffe", Index: 24},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "giraffe", Index: 23},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "giraffe", Index: 24},
	},
	"backpack": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "backpack", Index: 25},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "backpack", Index: 24},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "backpack", Index: 25},
	},
	"umbrella": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "umbrella", Index: 26},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "umbrella", Index: 25},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "umbrella", Index: 26},
	},
	"handbag": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "handbag", Index: 27},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "handbag", Index: 26},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "handbag", Index: 27},
	},
	"tie": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "tie", Index: 28},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "tie", Index: 27},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "tie", Index: 28},
	},
	"suitcase": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "suitcase", Index: 29},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "suitcase", Index: 28},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "suitcase", Index: 29},
	},
	"frisbee": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "frisbee", Index: 30},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "frisbee", Index: 29},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "frisbee", Index: 30},
	},
	"skis": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "skis", Index: 31},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "skis", Index: 30},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "skis", Index: 31},
	},
	"snowboard": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "snowboard", Index: 32},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "snowboard", Index: 31},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "snowboard", Index: 32},
	},
	"sports ball": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "sports ball", Index: 33},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "sports ball", Index: 32},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "sports ball", Index: 33},
	},
	"kite": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "kite", Index: 34},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "kite", Index: 33},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "kite", Index: 34},
	},
	"baseball bat": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "baseball bat", Index: 35},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "baseball bat", Index: 34},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "baseball bat", Index: 35},
	},
	"baseball glove": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "baseball glove", Index: 36},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "baseball glove", Index: 35},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "baseball glove", Index: 36},
	},
	"skateboard": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "skateboard", Index: 37},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "skateboard", Index: 36},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "skateboard", Index: 37},
	},
	"surfboard": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "surfboard", Index: 38},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "surfboard", Index: 37},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "surfboard", Index: 38},
	},
	"tennis racket": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "tennis racket", Index: 39},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "tennis racket", Index: 38},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "tennis racket", Index: 39},
	},
	"bottle": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bottle", Index: 40},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bottle", Index: 39},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bottle", Index: 40},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "bottle", Index: 5},
	},
	"wine glass": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "wine glass", Index: 41},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "wine glass", Index: 40},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "wine glass", Index: 41},
	},
	"cup": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "cup", Index: 42},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "cup", Index: 41},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "cup", Index: 42},
	},
	"fork": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "fork", Index: 43},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "fork", Index: 42},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "fork", Index: 43},
	},
	"knife": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "knife", Index: 44},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "knife", Index: 43},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "knife", Index: 44},
	},
	"spoon": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "spoon", Index: 45},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "spoon", Index: 44},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "spoon", Index: 45},
	},
	"bowl": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bowl", Index: 46},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bowl", Index: 45},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bowl", Index: 46},
	},
	"banana": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "banana", Index: 47},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "banana", Index: 46},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "banana", Index: 47},
	},
	"apple": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "apple", Index: 48},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "apple", Index: 47},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "apple", Index: 48},
	},
	"sandwich": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "sandwich", Index: 49},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "sandwich", Index: 48},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "sandwich", Index: 49},
	},
	"orange": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "orange", Index: 50},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "orange", Index: 49},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "orange", Index: 50},
	},
	"broccoli": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "broccoli", Index: 51},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "broccoli", Index: 50},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "broccoli", Index: 51},
	},
	"carrot": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "carrot", Index: 52},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "carrot", Index: 51},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "carrot", Index: 52},
	},
	"hot dog": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "hot dog", Index: 53},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "hot dog", Index: 52},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "hot dog", Index: 53},
	},
	"pizza": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "pizza", Index: 54},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "pizza", Index: 53},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "pizza", Index: 54},
	},
	"donut": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "donut", Index: 55},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "donut", Index: 54},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "donut", Index: 55},
	},
	"cake": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "cake", Index: 56},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "cake", Index: 55},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "cake", Index: 56},
	},
	"chair": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "chair", Index: 57},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "chair", Index: 56},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "chair", Index: 57},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "chair", Index: 9},
	},
	"couch": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "couch", Index: 58},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "couch", Index: 57},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "couch", Index: 58},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "sofa", Index: 18},
	},
	"sofa": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "couch", Index: 58},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "couch", Index: 57},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "couch", Index: 58},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "sofa", Index: 18},
	},
	"potted plant": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "potted plant", Index: 59},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "potted plant", Index: 58},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "potted plant", Index: 59},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "pottedplant", Index: 16},
	},
	"pottedplant": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "potted plant", Index: 59},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "potted plant", Index: 58},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "potted plant", Index: 59},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "pottedplant", Index: 16},
	},
	"bed": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "bed", Index: 60},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "bed", Index: 59},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "bed", Index: 60},
	},
	"dining table": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "dining table", Index: 61},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "dining table", Index: 60},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "dining table", Index: 61},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "diningtable", Index: 11},
	},
	"diningtable": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "dining table", Index: 61},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "dining table", Index: 60},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "dining table", Index: 61},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "diningtable", Index: 11},
	},
	"toilet": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "toilet", Index: 62},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "toilet", Index: 61},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "toilet", Index: 62},
	},
	"tv": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "tv", Index: 63},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "tv", Index: 62},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "tv", Index: 63},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "tvmonitor", Index: 20},
	},
	"tvmonitor": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "tv", Index: 63},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "tv", Index: 62},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "tv", Index: 63},
		model.ModelFamilyVOC:  {Family: model.ModelFamilyVOC, Name: "tvmonitor", Index: 20},
	},
	"laptop": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "laptop", Index: 64},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "laptop", Index: 63},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "laptop", Index: 64},
	},
	"mouse": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "mouse", Index: 65},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "mouse", Index: 64},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "mouse", Index: 65},
	},
	"remote": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "remote", Index: 66},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "remote", Index: 65},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "remote", Index: 66},
	},
	"keyboard": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "keyboard", Index: 67},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "keyboard", Index: 66},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "keyboard", Index: 67},
	},
	"cell phone": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "cell phone", Index: 68},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "cell phone", Index: 67},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "cell phone", Index: 68},
	},
	"microwave": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "microwave", Index: 69},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "microwave", Index: 68},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "microwave", Index: 69},
	},
	"oven": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "oven", Index: 70},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "oven", Index: 69},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "oven", Index: 70},
	},
	"toaster": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "toaster", Index: 71},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "toaster", Index: 70},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "toaster", Index: 71},
	},
	"sink": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "sink", Index: 72},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "sink", Index: 71},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "sink", Index: 72},
	},
	"refrigerator": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "refrigerator", Index: 73},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "refrigerator", Index: 72},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "refrigerator", Index: 73},
	},
	"book": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "book", Index: 74},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "book", Index: 73},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "book", Index: 74},
	},
	"clock": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "clock", Index: 75},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "clock", Index: 74},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "clock", Index: 75},
	},
	"vase": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "vase", Index: 76},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "vase", Index: 75},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "vase", Index: 76},
	},
	"scissors": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "scissors", Index: 77},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "scissors", Index: 76},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "scissors", Index: 77},
	},
	"teddy bear": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "teddy bear", Index: 78},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "teddy bear", Index: 77},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "teddy bear", Index: 78},
	},
	"hair drier": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "hair drier", Index: 79},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "hair drier", Index: 78},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "hair drier", Index: 79},
	},
	"toothbrush": {
		model.ModelFamilyCOCO: {Family: model.ModelFamilyCOCO, Name: "toothbrush", Index: 80},
		model.ModelFamilyYOLO: {Family: model.ModelFamilyYOLO, Name: "toothbrush", Index: 79},
		model.ModelFamilyTF:   {Family: model.ModelFamilyTF, Name: "toothbrush", Index: 80},
	},
}

// GetClass returns the class mapping for a given class name and returns an error
// if the class is not found.
//
// Arguments:
//   - name: The name of the class.
//
// Returns:
//   - map[model.Family]Class: The class mapping.
//   - error: An error if the class is not found.
func GetClass(name ClassName) (map[model.Family]Class, error) {
	if class, ok := Classes[name]; ok {
		return class, nil
	}
	return nil, fmt.Errorf("class %q not found", name)
}

// GetClasses returns the class mapping for a given class name.
//
// Arguments:
//   - name: The name of the class.
//
// Returns:
//   - map[model.Family]Class: The class mapping.
//   - error: An error if the class is not found.
func GetClasses(name ...ClassName) ([]Class, error) {
	classes := make([]Class, 0)
	for _, name := range name {
		class, err := GetClass(name)
		if err != nil {
			return nil, err
		}
		for _, class := range class {
			classes = append(classes, class)
		}
	}
	return classes, nil
}

// GetClassMapping returns the class mapping for a given class name and family
// and returns an error if the class is not found for the family.
//
// Arguments:
//   - name: The name of the class.
//   - family: The family of the model.
//
// Returns:
//   - ClassMapping: The class mapping.
//   - error: An error if the class is not found for the family.
func GetClassMapping(name ClassName, family model.Family) (Class, error) {
	mapping, ok := Classes[name][family]
	if !ok {
		return Class{}, fmt.Errorf("class %q not found for family %q", name, family)
	}
	return mapping, nil
}
