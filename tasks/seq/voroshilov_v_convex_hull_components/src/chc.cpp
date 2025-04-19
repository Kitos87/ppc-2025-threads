#include "../include/chc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>

using namespace voroshilov_v_convex_hull_components_seq;

Pixel::Pixel(int y_param, int x_param) : y(y_param), x(x_param), value(0) {}
Pixel::Pixel(int y_param, int x_param, int value_param) : y(y_param), x(x_param), value(value_param) {}

bool Pixel::operator==(const int value_param) const { return value == value_param; }
bool Pixel::operator==(const Pixel& other) const { return (y == other.y) && (x == other.x); }

Image::Image(int hght, int wdth, std::vector<int> pxls) {
  height = hght;
  width = wdth;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      Pixel pixel(y, x, pxls[(y * width) + x]);
      pixels.push_back(pixel);
    }
  }
}

Pixel& Image::GetPixel(int y, int x) { return pixels[(y * width) + x]; }

void Component::AddPixel(const Pixel& pixel) { pixels.push_back(pixel); }

LineSegment::LineSegment(Pixel& a_param, Pixel& b_param) : a(a_param), b(b_param) {}

bool Hull::operator==(const Hull& other) const { return pixels == other.pixels; }

Component voroshilov_v_convex_hull_components_seq::DepthComponentSearch(Pixel& start_pixel, Image* tmp_image,
                                                                        int index) {
  const int step_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int step_x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)
  std::stack<Pixel> stack;
  Component component;
  stack.push(start_pixel);
  tmp_image->GetPixel(start_pixel.y, start_pixel.x).value = index;        // Mark start pixel as visited
  component.AddPixel(tmp_image->GetPixel(start_pixel.y, start_pixel.x));  // Add start pixel to component

  while (!stack.empty()) {
    Pixel current_pixel = stack.top();
    stack.pop();
    for (int i = 0; i < 8; i++) {
      int next_y = current_pixel.y + step_y[i];
      int next_x = current_pixel.x + step_x[i];
      if (next_y >= 0 && next_y < tmp_image->height && next_x >= 0 && next_x < tmp_image->width &&
          tmp_image->GetPixel(next_y, next_x) == 1) {
        stack.push(tmp_image->GetPixel(next_y, next_x));
        tmp_image->GetPixel(next_y, next_x).value = index;        // Mark neighbour pixel as visited
        component.AddPixel(tmp_image->GetPixel(next_y, next_x));  // Add neighbour pixel to component
      }
    }
  }

  return component;
}

std::vector<Component> voroshilov_v_convex_hull_components_seq::FindComponents(Image& image) {
  Image tmp_image(image);
  std::vector<Component> components;
  int count = 0;
  for (int y = 0; y < tmp_image.height; y++) {
    for (int x = 0; x < tmp_image.width; x++) {
      if (tmp_image.GetPixel(y, x) == 1) {
        Component component = DepthComponentSearch(tmp_image.GetPixel(y, x), &tmp_image, count + 2);
        components.push_back(component);
        count++;
      }
    }
  }
  if (components.empty()) {
    return {};
  }
  return components;
}

int voroshilov_v_convex_hull_components_seq::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  return ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));
}

Pixel voroshilov_v_convex_hull_components_seq::FindFarthestPixel(std::vector<Pixel>& pixels,
                                                                 LineSegment& line_segment) {
  Pixel farthest_pixel(-1, -1, -1);
  double max_dist = 0.0;

  for (Pixel& c : pixels) {
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    if (CheckRotation(a, b, c) < 0) {  // left rotation
      double distance = std::abs(((b.x - a.x) * (a.y - c.y)) - ((a.x - c.x) * (b.y - a.y)));
      if (distance > max_dist) {
        max_dist = distance;
        farthest_pixel = c;
      }
    }
  }

  return farthest_pixel;
}

std::vector<Pixel> voroshilov_v_convex_hull_components_seq::QuickHull(Component& component) {
  if (component.pixels.size() < 3) {
    return component.pixels;
  }

  Pixel left = component.pixels[0];
  Pixel right = component.pixels[0];

  for (Pixel& pixel : component.pixels) {
    if (pixel.x < left.x) {
      left = pixel;
    }
    if (pixel.x > right.x) {
      right = pixel;
    }
  }

  std::vector<Pixel> hull;
  std::stack<LineSegment> stack;

  LineSegment line_segment1(left, right);
  LineSegment line_segment2(right, left);
  stack.push(line_segment1);
  stack.push(line_segment2);

  while (!stack.empty()) {
    LineSegment line_segment = stack.top();
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    stack.pop();

    Pixel c = FindFarthestPixel(component.pixels, line_segment);
    if (c == -1) {
      hull.push_back(a);
    } else {
      LineSegment new_line1(a, c);
      stack.push(new_line1);
      LineSegment new_line2(c, b);
      stack.push(new_line2);
    }
  }

  std::ranges::reverse(hull);

  std::vector<Pixel> res_hull;
  for (size_t i = 0; i < hull.size(); i++) {
    if (i == 0 || i == hull.size() - 1 || CheckRotation(hull[i - 1], hull[i], hull[i + 1]) != 0) {
      res_hull.push_back(hull[i]);
    }
  }

  return res_hull;
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::QuickHullAll(std::vector<Component>& components) {
  if (components.empty()) {
    return {};
  }
  std::vector<Hull> hulls;
  for (Component& component : components) {
    Hull hull;
    hull.pixels = QuickHull(component);
    hulls.push_back(hull);
  }
  return hulls;
}

std::pair<std::vector<int>, std::vector<int>> voroshilov_v_convex_hull_components_seq::PackHulls(
    std::vector<Hull>& hulls, Image& image) {
  int height = image.height;
  int width = image.width;

  std::vector<int> hulls_indexes(height * width, 0);
  std::vector<int> pixels_indexes(height * width, 0);

  int hull_index = 1;
  for (Hull& hull : hulls) {
    int pixel_index = 1;
    for (Pixel& pixel : hull.pixels) {
      hulls_indexes[(pixel.y * width) + pixel.x] = hull_index;
      pixels_indexes[(pixel.y * width) + pixel.x] = pixel_index;
      pixel_index++;
    }
    hull_index++;
  }

  std::pair<std::vector<int>, std::vector<int>> packed_vectors(hulls_indexes, pixels_indexes);
  return packed_vectors;
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::UnpackHulls(std::vector<int>& hulls_indexes,
                                                                       std::vector<int>& pixels_indexes, int height,
                                                                       int width, size_t hulls_size) {
  std::vector<Hull> hulls(hulls_size);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int hull_index = hulls_indexes[(y * width) + x];
      if (hull_index > 0) {
        int pixel_index = pixels_indexes[(y * width) + x];
        Pixel pixel(y, x, pixel_index);
        hulls[hull_index - 1].pixels.push_back(pixel);
      }
    }
  }

  for (Hull& hull : hulls) {
    for (size_t p1 = 0; p1 < hull.pixels.size() - 1; p1++) {
      for (size_t p2 = p1 + 1; p2 < hull.pixels.size(); p2++) {
        if (hull.pixels[p1].value > hull.pixels[p2].value) {
          Pixel tmp = hull.pixels[p1];
          hull.pixels[p1] = hull.pixels[p2];
          hull.pixels[p2] = tmp;
        }
      }
    }
  }

  return hulls;
}
