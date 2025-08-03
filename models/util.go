package models

// import (
// 	"errors"
// 	"sync/atomic"
// 	"time"
// )

// type ModelServer struct {
// 	currentModel atomic.Value
// 	newModel     atomic.Value
// 	switching    int32
// }

// func (ms *ModelServer) UpdateModel(newModel Model) error {
// 	ms.newModel.Store(newModel)
// 	if atomic.CompareAndSwapInt32(&ms.switching, 0, 1) {
// 		defer atomic.StoreInt32(&ms.switching, 0)

// 		time.Sleep(100 * time.Millisecond) // Wait for current requests
// 		ms.currentModel.Store(ms.newModel.Load())
// 		return nil
// 	}
// 	return errors.New("update in progress")
// }
