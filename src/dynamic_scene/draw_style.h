#ifndef CMU462_DYNAMICSCENE_DRAWSTYLE_H
#define CMU462_DYNAMICSCENE_DRAWSTYLE_H

#include "scene.h"

namespace CMU462 { namespace DynamicScene {

/**
 * Used in rendering to specify how to draw the faces/meshes/lines/etc.
 */
class DrawStyle {
 public:
  void style_face() const {
    glColor3f(faceColor.r, faceColor.g, faceColor.b);
  }

  void style_edge() const {
    glColor3f(edgeColor.r, edgeColor.g, edgeColor.b);
    glLineWidth(strokeWidth);
  }

  void style_halfedge() const {
    glColor3f(halfedgeColor.r, halfedgeColor.g, halfedgeColor.b);
    glLineWidth(strokeWidth);
  }

  void style_vertex() const {
    glColor3f(vertexColor.r, vertexColor.g, vertexColor.b);
    glPointSize(vertexRadius);
  }

  Color halfedgeColor;
  Color vertexColor;
  Color edgeColor;
  Color faceColor;

  float strokeWidth;
  float vertexRadius;
};

} // namespace DynamicScene
} // namespace CMU462

#endif //CMU462_DYNAMICSCENE_DRAWSTYLE_H
