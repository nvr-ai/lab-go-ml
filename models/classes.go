package models

import "fmt"

// OutputClass represents one detection label.
type OutputClass struct {
	// The integer index returned by the model.
	Index int
	// The human-readable label.
	Name string
}

// OutputClassSet ties a style to its full list of labels.
type OutputClassSet struct {
	// Class set identifier.
	Style ModelFamily
	// Classes that are supported and mappable.
	Classes []OutputClass
	// nameToIdx for fast lookup by name
	nameToIdx map[string]int
}

// BuildNameIndexMap builds or rebuilds the name->index map.
func (s *OutputClassSet) BuildNameIndexMap() {
	s.nameToIdx = make(map[string]int, len(s.Classes))
	for _, c := range s.Classes {
		s.nameToIdx[c.Name] = c.Index
	}
}

// ClassManager holds all registered class sets.
type ClassManager struct {
	sets map[ModelFamily]*OutputClassSet
}

// NewClassManager initializes and registers the given sets.
func NewClassManager(allSets ...*OutputClassSet) *ClassManager {
	mgr := &ClassManager{sets: make(map[ModelFamily]*OutputClassSet)}
	for _, set := range allSets {
		set.BuildNameIndexMap()
		mgr.sets[set.Style] = set
	}
	return mgr
}

// GetName returns the class name for a given style and index.
func (m *ClassManager) GetName(style ModelFamily, idx int) (string, error) {
	set, ok := m.sets[style]
	if !ok {
		return "", fmt.Errorf("style %q not registered", style)
	}
	if idx < 0 || idx >= len(set.Classes) {
		return "", fmt.Errorf("index %d out of range for style %q", idx, style)
	}
	return set.Classes[idx].Name, nil
}

// GetIndex returns the class index for a given style and name.
func (m *ClassManager) GetIndex(style ModelFamily, name string) (int, error) {
	set, ok := m.sets[style]
	if !ok {
		return -1, fmt.Errorf("style %q not registered", style)
	}
	idx, ok := set.nameToIdx[name]
	if !ok {
		return -1, fmt.Errorf("name %q not found in style %q", name, style)
	}
	return idx, nil
}

// MapClass maps an index from one style to another, returning the target OutputClass.
func (m *ClassManager) MapClass(fromStyle ModelFamily, idx int, toStyle ModelFamily) (OutputClass, error) {
	name, err := m.GetName(fromStyle, idx)
	if err != nil {
		return OutputClass{}, err
	}
	toIdx, err := m.GetIndex(toStyle, name)
	if err != nil {
		return OutputClass{}, err
	}
	return OutputClass{Index: toIdx, Name: name}, nil
}

// COCOClasses is the full 80 COCO classes plus "__background__" at index 0.
var COCOClasses = OutputClassSet{
	Style: ModelFamilyCOCO,
	Classes: []OutputClass{
		{0, "__background__"},
		{1, "person"},
		{2, "bicycle"},
		{3, "car"},
		{4, "motorcycle"},
		{5, "airplane"},
		{6, "bus"},
		{7, "train"},
		{8, "truck"},
		{9, "boat"},
		{10, "traffic light"},
		{11, "fire hydrant"},
		{12, "stop sign"},
		{13, "parking meter"},
		{14, "bench"},
		{15, "bird"},
		{16, "cat"},
		{17, "dog"},
		{18, "horse"},
		{19, "sheep"},
		{20, "cow"},
		{21, "elephant"},
		{22, "bear"},
		{23, "zebra"},
		{24, "giraffe"},
		{25, "backpack"},
		{26, "umbrella"},
		{27, "handbag"},
		{28, "tie"},
		{29, "suitcase"},
		{30, "frisbee"},
		{31, "skis"},
		{32, "snowboard"},
		{33, "sports ball"},
		{34, "kite"},
		{35, "baseball bat"},
		{36, "baseball glove"},
		{37, "skateboard"},
		{38, "surfboard"},
		{39, "tennis racket"},
		{40, "bottle"},
		{41, "wine glass"},
		{42, "cup"},
		{43, "fork"},
		{44, "knife"},
		{45, "spoon"},
		{46, "bowl"},
		{47, "banana"},
		{48, "apple"},
		{49, "sandwich"},
		{50, "orange"},
		{51, "broccoli"},
		{52, "carrot"},
		{53, "hot dog"},
		{54, "pizza"},
		{55, "donut"},
		{56, "cake"},
		{57, "chair"},
		{58, "couch"},
		{59, "potted plant"},
		{60, "bed"},
		{61, "dining table"},
		{62, "toilet"},
		{63, "tv"},
		{64, "laptop"},
		{65, "mouse"},
		{66, "remote"},
		{67, "keyboard"},
		{68, "cell phone"},
		{69, "microwave"},
		{70, "oven"},
		{71, "toaster"},
		{72, "sink"},
		{73, "refrigerator"},
		{74, "book"},
		{75, "clock"},
		{76, "vase"},
		{77, "scissors"},
		{78, "teddy bear"},
		{79, "hair drier"},
		{80, "toothbrush"},
	},
}

// YOLOClasses is the 80 COCO classes (no background).
// YOLO models index directly into this zero-based list.
var YOLOClasses = OutputClassSet{
	Style: ModelFamilyYOLO,
	Classes: func() []OutputClass {
		classes := make([]OutputClass, len(COCOClasses.Classes)-1) // drop background
		for i := 1; i < len(COCOClasses.Classes); i++ {
			classes[i-1] = OutputClass{i - 1, COCOClasses.Classes[i].Name}
		}
		return classes
	}(),
}

// TFCOCOClasses mirrors TensorFlow’s default COCO labelmap (80 + background).
// Use these indices when running TF-exported models in ONNX Runtime GO.
var TFCOCOClasses = OutputClassSet{
	Style:   ModelFamilyTF,
	Classes: COCOClasses.Classes, // identical names & indices
}

// PascalVOCClasses is the 20 Pascal VOC classes + "__background__" at index 0.
var PascalVOCClasses = OutputClassSet{
	Style: ModelFamilyVOC,
	Classes: []OutputClass{
		{0, "__background__"},
		{1, "aeroplane"},
		{2, "bicycle"},
		{3, "bird"},
		{4, "boat"},
		{5, "bottle"},
		{6, "bus"},
		{7, "car"},
		{8, "cat"},
		{9, "chair"},
		{10, "cow"},
		{11, "diningtable"},
		{12, "dog"},
		{13, "horse"},
		{14, "motorbike"},
		{15, "person"},
		{16, "pottedplant"},
		{17, "sheep"},
		{18, "sofa"},
		{19, "train"},
		{20, "tvmonitor"},
	},
}

// AllClassSets collects every OutputClassSet in one place.
// Helps you iterate across all supported label mappings.
var AllClassSets = []OutputClassSet{
	COCOClasses,
	YOLOClasses,
	TFCOCOClasses,
	PascalVOCClasses,
}

// LookupName returns the class name for a given style and index.
// If index is out of range, it returns an empty string.
func LookupName(style ModelFamily, idx int) string {
	for _, set := range AllClassSets {
		if set.Style == style {
			if idx >= 0 && idx < len(set.Classes) {
				return set.Classes[idx].Name
			}
			return ""
		}
	}
	return ""
}
