#pragma once

#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace miter {

/**
 * @brief Contiguous heap buffer aligned to Alignment bytes (default 64 for cache lines/SIMD).
 */
template <typename T, std::size_t Alignment = 64> class AlignedBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "AlignedBuffer requires trivially copyable T");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

public:
    AlignedBuffer() noexcept = default;

    explicit AlignedBuffer(std::size_t n) { allocate(n); }

    ~AlignedBuffer() { deallocate(); }

    AlignedBuffer(const AlignedBuffer& other) {
        if (other.size_ > 0) {
            allocate(other.size_);
            std::memcpy(ptr_, other.ptr_, size_ * sizeof(T));
        }
    }

    AlignedBuffer& operator=(const AlignedBuffer& other) {
        if (this == &other) {
            return *this;
        }
        AlignedBuffer tmp(other);
        swap(tmp);
        return *this;
    }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        deallocate();
        ptr_ = std::exchange(other.ptr_, nullptr);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    void resize(std::size_t n) {
        if (n == size_) {
            return;
        }
        deallocate();
        allocate(n);
    }

    void zero_fill() noexcept {
        if (size_ > 0) {
            std::memset(static_cast<void*>(ptr_), 0, size_ * sizeof(T));
        }
    }

    void swap(AlignedBuffer& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(size_, other.size_);
    }

    T* data() noexcept { return ptr_; }

    const T* data() const noexcept { return ptr_; }

    std::size_t size() const noexcept { return size_; }

    bool empty() const noexcept { return size_ == 0; }

    T& operator[](std::size_t idx) noexcept { return ptr_[idx]; }

    const T& operator[](std::size_t idx) const noexcept { return ptr_[idx]; }

    T* begin() noexcept { return ptr_; }

    T* end() noexcept { return ptr_ + size_; }

    const T* begin() const noexcept { return ptr_; }

    const T* end() const noexcept { return ptr_ + size_; }

private:
    void allocate(std::size_t n) {
        if (n == 0) {
            ptr_ = nullptr;
            size_ = 0;
            return;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        ptr_ = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
        size_ = n;
    }

    void deallocate() noexcept {
        if (ptr_ != nullptr) {
            ::operator delete(ptr_, std::align_val_t{Alignment});
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace miter
