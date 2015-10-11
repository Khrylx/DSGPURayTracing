#ifndef CMU462_IMAGE_H
#define CMU462_IMAGE_H

#include "CMU462/color.h"

namespace CMU462 {

/**
 * Represents a mutable image which uses 8-bit RGBA pixel layout.
 */
class ImageBuffer {
 public:
  /**
   * Default constructor.
   * The default constructor creates a zero-sized image.
   */
  ImageBuffer() : w(0), h(0) {}

  /**
   * Parameterized constructor.
   * Create an image of given size.
   * \param w width of the image
   * \param h height of the image
   */
  ImageBuffer(size_t w, size_t h) : w(w), h(h) { buffer.resize(4 * w * h); }

  /**
   * Get width of the image.
   * \return width of the image.
   */
  size_t width() const { return w; }

  /**
   * Get height of the image.
   * \return width of the image.
   */
  size_t height() const { return h; }

  /**
   * Get address of pixels of the image.
   * \return pointer to the first pixel of the image.
   */
  const uint8_t* pixels() const { return buffer.data(); }

  /**
   * Update the color of a given pixel.
   * \param c color value to be set
   * \param x row of the pixel
   * \param y column of the pixel
   */
  void put_color(const Color& c, size_t x, size_t y) {

    // assert(0 <= x && x < w);
    // assert(0 <= y && y < h);

    buffer[4 * (y * w + x)] = clamp(0.f, 1.f, c.r) * 255;
    buffer[4 * (y * w + x) + 1] = clamp(0.f, 1.f, c.g) * 255;
    buffer[4 * (y * w + x) + 2] = clamp(0.f, 1.f, c.b) * 255;
    buffer[4 * (y * w + x) + 3] = clamp(0.f, 1.f, c.a) * 255;
  }

  /**
   * Resize the image buffer.
   * \param w new width of the image
   * \param h new height of the image
   */
  void resize(size_t w, size_t h) {
    this->w = w;
    this->h = h;
    buffer.resize(4 * w * h);
    clear();
  }

  /**
   * Fills the buffer with zeros (black pixels).
   */
  void clear() {
    for (int i = 0; i < 4 * w * h; i += 4) {
      buffer[i] = buffer[i + 1] = buffer[i + 2] = 0;
      buffer[i + 3] = 255;
    }
  }

  bool is_empty() { return w == 0 || h == 0; }

 private:
  size_t w;                ///< width
  size_t h;                ///< height
  vector<uint8_t> buffer;  ///< pixel buffer
};

} // namespace CMU462

#endif // CMU462_IMAGE_H
