/*
 * MVKSync.mm
 *
 * Copyright (c) 2014-2019 The Brenwill Workshop Ltd. (http://www.brenwill.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MVKSync.h"
#include "MVKFoundation.h"
#include "MVKLogging.h"

using namespace std;


#pragma mark -
#pragma mark MVKSemaphoreImpl

bool MVKSemaphoreImpl::release() {
	lock_guard<mutex> lock(_lock);
    if (isClear()) { return true; }

    // Either decrement the reservation counter, or clear it altogether
    if (_shouldWaitAll) {
		if (_reservationCount > 0) { _reservationCount--; }
    } else {
        _reservationCount = 0;
    }
    // If all reservations have been released, unblock all waiting threads
    if ( isClear() ) { _blocker.notify_all(); }
    return isClear();
}

void MVKSemaphoreImpl::reserve() {
	lock_guard<mutex> lock(_lock);
	_reservationCount++;
}

bool MVKSemaphoreImpl::isReserved() {
	lock_guard<mutex> lock(_lock);
	return !isClear();
}

bool MVKSemaphoreImpl::wait(uint64_t timeout, bool reserveAgain) {
    unique_lock<mutex> lock(_lock);

    bool isDone;
    if (timeout == 0) {
		isDone = isClear();
	} else if (timeout == UINT64_MAX) {
		_blocker.wait(lock, [this]{ return isClear(); });
		isDone = true;
	} else {
        // Limit timeout to avoid overflow since wait_for() uses wait_until()
        uint64_t nanoTimeout = min(timeout, kMVKUndefinedLargeUInt64);
        chrono::nanoseconds nanos(nanoTimeout);
        isDone = _blocker.wait_for(lock, nanos, [this]{ return isClear(); });
    }

	if (reserveAgain) { _reservationCount++; }
    return isDone;
}

MVKSemaphoreImpl::~MVKSemaphoreImpl() {
    // Acquire the lock to ensure proper ordering.
    lock_guard<mutex> lock(_lock);
}


#pragma mark -
#pragma mark MVKSemaphoreMTLFence

// Could use any encoder. Assume BLIT is fastest and lightest.
// Nil mtlCmdBuff will do nothing.
VkResult MVKSemaphoreMTLFence::encodeWait(id<MTLCommandBuffer> mtlCmdBuff) {
	id<MTLBlitCommandEncoder> mtlCmdEnc = mtlCmdBuff.blitCommandEncoder;
	[mtlCmdEnc waitForFence: _mtlFence];
	[mtlCmdEnc endEncoding];
	return VK_SUCCESS;
}

// Could use any encoder. Assume BLIT is fastest and lightest.
// Nil mtlCmdBuff will do nothing.
VkResult MVKSemaphoreMTLFence::encodeSignal(id<MTLCommandBuffer> mtlCmdBuff) {
	id<MTLBlitCommandEncoder> mtlCmdEnc = mtlCmdBuff.blitCommandEncoder;
	[mtlCmdEnc updateFence: _mtlFence];
	[mtlCmdEnc endEncoding];
	return VK_SUCCESS;
}

MVKSemaphoreMTLFence::MVKSemaphoreMTLFence(MVKDevice* device, const VkSemaphoreCreateInfo* pCreateInfo) :
	MVKSemaphore(device, pCreateInfo),
	_mtlFence([device->getMTLDevice() newFence]) {}		//retained

MVKSemaphoreMTLFence::~MVKSemaphoreMTLFence() {
	[_mtlFence release];
}


#pragma mark -
#pragma mark MVKSemaphoreMTLEvent

// Nil mtlCmdBuff will do nothing.
VkResult MVKSemaphoreMTLEvent::encodeWait(id<MTLCommandBuffer> mtlCmdBuff) {
	if (mtlCmdBuff)
	{
		// check there is only one signal ahead
		if (_mtlSignalEventValue != _mtlWaitEventValue + 1) return reportError(VK_ERROR_DEVICE_LOST, "VkSemaphore must be signaled or have a pending signal operation before queueing a wait operation");

		[mtlCmdBuff encodeWaitForEvent: _mtlEvent value: _mtlWaitEventValue++];
	}
	return VK_SUCCESS;
}

// Nil mtlCmdBuff will do nothing.
VkResult MVKSemaphoreMTLEvent::encodeSignal(id<MTLCommandBuffer> mtlCmdBuff) {
	if (mtlCmdBuff)
	{
		// check there hasn't been wait without signal
		if (_mtlSignalEventValue != _mtlWaitEventValue) return reportError(VK_ERROR_DEVICE_LOST, "VkSemaphore must be unsignaled or have a pending wait operation before queueing a signal operation");

		[mtlCmdBuff encodeSignalEvent: _mtlEvent value: _mtlSignalEventValue++];
	}
	return VK_SUCCESS;
}

VkResult MVKSemaphoreMTLEvent::signalFromCPU() {
	// check there hasn't been wait without signal
	if (_mtlSignalEventValue != _mtlWaitEventValue) return reportError(VK_ERROR_DEVICE_LOST, "VkSemaphore must be unsignaled or have a pending wait operation before queueing a signal operation");

	// check that there is no pending signals previously submited but that haven't yet happened
	// vulkan requires that a semaphore can only be re-signaled when it can be proven
	// that both the previously submited signals and waits have completed by the point the new signal is executed
	// ie, signal and wait operations have to happen on the same order they are submited
	if (_mtlSignalEventValue - 1 != _mtlEvent.signaledValue) return reportError(VK_ERROR_DEVICE_LOST, "VkSemaphore signal operations must happen in the same order as they are queued (expected %u, got %u)", _mtlSignalEventValue - 1, (uint32_t)_mtlEvent.signaledValue);

	_mtlEvent.signaledValue = _mtlSignalEventValue++;

	return VK_SUCCESS;
}

VkResult MVKSemaphoreMTLEvent::delayedSignalFromCPU(uint32_t& outSignalValue) {
	// check there hasn't been wait without signal
	if (_mtlSignalEventValue != _mtlWaitEventValue) return reportError(VK_ERROR_DEVICE_LOST, "VkSemaphore must be unsignaled or have a pending wait operation before queueing a signal operation");

	outSignalValue = _mtlSignalEventValue++;

	return VK_SUCCESS;
}

VkResult MVKSemaphoreMTLEvent::commitDelayedSignalFromCPU(uint32_t signalValue) {
	// check that there is no pending signals previously submited but that haven't yet happened
	// vulkan requires that a semaphore can only be re-signaled when it can be proven
	// that both the previously submited signals and waits have completed by the point the new signal is executed
	// ie, signal and wait operations have to happen on the same order they are submited
	if (signalValue - 1 != _mtlEvent.signaledValue) return reportError(VK_ERROR_DEVICE_LOST, "VkSemaphore signal operations must happen in the same order as they are queued (expected %u, got %u)", signalValue - 1, (uint32_t)_mtlEvent.signaledValue);

	_mtlEvent.signaledValue = signalValue;

	return VK_SUCCESS;
}

MVKSemaphoreMTLEvent::MVKSemaphoreMTLEvent(MVKDevice* device, const VkSemaphoreCreateInfo* pCreateInfo) :
	MVKSemaphore(device, pCreateInfo),
	_mtlEvent([device->getMTLDevice() newSharedEvent]),	//retained
	_mtlWaitEventValue(1),
	_mtlSignalEventValue(1) {}

MVKSemaphoreMTLEvent::~MVKSemaphoreMTLEvent() {
    [_mtlEvent release];
}


#pragma mark -
#pragma mark MVKSemaphoreEmulated

VkResult MVKSemaphoreEmulated::encodeWait(id<MTLCommandBuffer> mtlCmdBuff) {
	if ( !mtlCmdBuff ) { _blocker.wait(UINT64_MAX, true); }
	return VK_SUCCESS;
}

VkResult MVKSemaphoreEmulated::encodeSignal(id<MTLCommandBuffer> mtlCmdBuff) {
	if ( !mtlCmdBuff ) { _blocker.release(); }
	return VK_SUCCESS;
}

MVKSemaphoreEmulated::MVKSemaphoreEmulated(MVKDevice* device, const VkSemaphoreCreateInfo* pCreateInfo) :
	MVKSemaphore(device, pCreateInfo),
	_blocker(false, 1) {}


#pragma mark -
#pragma mark MVKFence

void MVKFence::addSitter(MVKFenceSitter* fenceSitter) {
	lock_guard<mutex> lock(_lock);

	// We only care about unsignaled fences. If already signaled,
	// don't add myself to the sitter and don't signal the sitter.
	if (_isSignaled) { return; }

	// Ensure each fence only added once to each fence sitter
	auto addRslt = _fenceSitters.insert(fenceSitter);	// pair with second element true if was added
	if (addRslt.second) { fenceSitter->awaitFence(this); }
}

void MVKFence::removeSitter(MVKFenceSitter* fenceSitter) {
	lock_guard<mutex> lock(_lock);

	_fenceSitters.erase(fenceSitter);
}

void MVKFence::signal() {
	lock_guard<mutex> lock(_lock);

	if (_isSignaled) { return; }	// Only signal once
	_isSignaled = true;

	// Notify all the fence sitters, and clear them from this instance.
    for (auto& fs : _fenceSitters) {
        fs->fenceSignaled(this);
    }
	_fenceSitters.clear();
}

void MVKFence::reset() {
	lock_guard<mutex> lock(_lock);

	_isSignaled = false;
	_fenceSitters.clear();
}

bool MVKFence::getIsSignaled() {
	lock_guard<mutex> lock(_lock);

	return _isSignaled;
}


#pragma mark -
#pragma mark MVKEventNative

// Odd == set / Even == reset.
bool MVKEventNative::isSet() { return _mtlEvent.signaledValue & 1; }

void MVKEventNative::signal(bool status) {
	if (isSet() != status) {
		_mtlEvent.signaledValue += 1;
	}
}

void MVKEventNative::encodeSignal(id<MTLCommandBuffer> mtlCmdBuff, bool status) {
	if (isSet() != status) {
		[mtlCmdBuff encodeSignalEvent: _mtlEvent value: _mtlEvent.signaledValue + 1];
	}
}

void MVKEventNative::encodeWait(id<MTLCommandBuffer> mtlCmdBuff) {
	if ( !isSet() ) {
		[mtlCmdBuff encodeWaitForEvent: _mtlEvent value: _mtlEvent.signaledValue + 1];
	}
}

MVKEventNative::MVKEventNative(MVKDevice* device, const VkEventCreateInfo* pCreateInfo) : MVKEvent(device, pCreateInfo) {
	_mtlEvent = [_device->getMTLDevice() newSharedEvent];	// retained
}

MVKEventNative::~MVKEventNative() {
	[_mtlEvent release];
}


#pragma mark -
#pragma mark MVKEventEmulated

bool MVKEventEmulated::isSet() { return !_blocker.isReserved(); }

void MVKEventEmulated::signal(bool status) {
	if (status) {
		_blocker.release();
	} else {
		_blocker.reserve();
	}
}

void MVKEventEmulated::encodeSignal(id<MTLCommandBuffer> mtlCmdBuff, bool status) {
	if (status) {
		[mtlCmdBuff addCompletedHandler: ^(id<MTLCommandBuffer> mcb) { _blocker.release(); }];
	} else {
		_blocker.reserve();
	}

	// An encoded signal followed by an encoded wait should cause the wait to be skipped.
	// However, because encoding a signal will not release the blocker until the command buffer
	// is finished executing (so the CPU can tell when it really is done) it is possible that
	// the encoded wait will block when it shouldn't. To avoid that, we keep track of whether
	// the most recent encoded signal was set or reset, so the next encoded wait knows whether
	// to really wait or not.
	_inlineSignalStatus = status;
}

void MVKEventEmulated::encodeWait(id<MTLCommandBuffer> mtlCmdBuff) {
	if ( !_inlineSignalStatus ) { _blocker.wait(); }
}

MVKEventEmulated::MVKEventEmulated(MVKDevice* device, const VkEventCreateInfo* pCreateInfo) :
	MVKEvent(device, pCreateInfo), _blocker(false, 1), _inlineSignalStatus(false) {}


#pragma mark -
#pragma mark Support functions

VkResult mvkResetFences(uint32_t fenceCount, const VkFence* pFences) {
	for (uint32_t i = 0; i < fenceCount; i++) {
		((MVKFence*)pFences[i])->reset();
	}
	return VK_SUCCESS;
}

// Create a blocking fence sitter, add it to each fence, wait, then remove it.
VkResult mvkWaitForFences(MVKDevice* device,
						  uint32_t fenceCount,
						  const VkFence* pFences,
						  VkBool32 waitAll,
						  uint64_t timeout) {

	VkResult rslt = VK_SUCCESS;
	MVKFenceSitter fenceSitter(waitAll);

	for (uint32_t i = 0; i < fenceCount; i++) {
		((MVKFence*)pFences[i])->addSitter(&fenceSitter);
	}

	if ( !fenceSitter.wait(timeout) ) { rslt = VK_TIMEOUT; }

	for (uint32_t i = 0; i < fenceCount; i++) {
		((MVKFence*)pFences[i])->removeSitter(&fenceSitter);
	}

	return rslt;
}


#pragma mark -
#pragma mark MVKMetalCompiler

// Create a compiled object by dispatching the block to the default global dispatch queue, and waiting only as long
// as the MVKConfiguration::metalCompileTimeout value. If the timeout is triggered, a Vulkan error is created.
// This approach is used to limit the lengthy time (30+ seconds!) consumed by Metal when it's internal compiler fails.
// The thread dispatch is needed because even the sync portion of the async Metal compilation methods can take well
// over a second to return when a compiler failure occurs!
void MVKMetalCompiler::compile(unique_lock<mutex>& lock, dispatch_block_t block) {
	MVKAssert( _startTime == 0, "%s compile occurred already in this instance. Instances of %s should only be used for a single compile activity.", _compilerType.c_str(), getClassName().c_str());

	MVKDevice* mvkDev = _owner->getDevice();
	_startTime = mvkDev->getPerformanceTimestamp();

	dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{ @autoreleasepool { block(); } });

	// Limit timeout to avoid overflow since wait_for() uses wait_until()
	chrono::nanoseconds nanoTimeout(min(mvkDev->_pMVKConfig->metalCompileTimeout, kMVKUndefinedLargeUInt64));
	_blocker.wait_for(lock, nanoTimeout, [this]{ return _isCompileDone; });

	if ( !_isCompileDone ) {
		@autoreleasepool {
			NSString* errDesc = [NSString stringWithFormat: @"Timeout after %.3f milliseconds. Likely internal Metal compiler error", (double)nanoTimeout.count() / 1e6];
			_compileError = [[NSError alloc] initWithDomain: @"MoltenVK" code: 1 userInfo: @{NSLocalizedDescriptionKey : errDesc}];	// retained
		}
	}

	if (_compileError) { handleError(); }

	mvkDev->addActivityPerformance(*_pPerformanceTracker, _startTime);
}

void MVKMetalCompiler::handleError() {
	_owner->setConfigurationResult(reportError(VK_ERROR_INITIALIZATION_FAILED,
											   "%s compile failed (Error code %li):\n%s.",
											   _compilerType.c_str(), (long)_compileError.code,
											   _compileError.localizedDescription.UTF8String));
}

// Returns whether the compilation came in late, after the compiler was destroyed.
bool MVKMetalCompiler::endCompile(NSError* compileError) {
	_compileError = [compileError retain];		// retained
	_isCompileDone = true;
	_blocker.notify_all();
	return _isDestroyed;
}

void MVKMetalCompiler::destroy() {
	if (markDestroyed()) { MVKBaseObject::destroy(); }
}

// Marks this object as destroyed, and returns whether the compilation is complete.
bool MVKMetalCompiler::markDestroyed() {
	lock_guard<mutex> lock(_completionLock);

	_isDestroyed = true;
	return _isCompileDone;
}


#pragma mark Construction

MVKMetalCompiler::~MVKMetalCompiler() {
	[_compileError release];
}



