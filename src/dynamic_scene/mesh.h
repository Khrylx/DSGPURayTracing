#ifndef CMU462_DYNAMICSCENE_MESH_H
#define CMU462_DYNAMICSCENE_MESH_H

#include "scene.h"

#include "../collada/polymesh_info.h"
#include "../halfEdgeMesh.h"
#include "../meshEdit.h"

namespace CMU462 { namespace DynamicScene {

/**
 * A MeshFeature is used to represent an element of the surface selected
 * by the user (e.g., edge, vertex, face).  No matter what kind of feature
 * is selected, the feature is specified relative to some polygon in the
 * mesh.  For instance, if an edge is selected, the MeshFeature will store
 * a pointer to a face containing that edge, as well as the local index of
 * the first (of two) vertices in the polygon corresponding to the edge.
 */
class MeshFeature {
 public:
  MeshFeature() : element(nullptr), w(0.0) { }

  bool isValid() const {
     return element != NULL;
  }

  void invalidate() {
    element = NULL;
  }

  Vector3D bCoords;         ///< Barycentric coordinates of selection
  HalfedgeElement* element; ///< element selected
  double w;                 ///< depth value of selection
};


class Mesh : public SceneObject, public MeshView {
 public:

  Mesh(Collada::PolymeshInfo& polyMesh, const Matrix4x4& transform);

  ~Mesh();

  void set_draw_styles(DrawStyle *defaultStyle, DrawStyle *hoveredStyle,
                       DrawStyle *selectedStyle);
  void render_in_opengl() const;

  BBox get_bbox();

  double test_selection(const Vector2D& p, const Matrix4x4& worldTo3DH,
                        double minW);

  void confirm_hover();
  void confirm_select();
  void invalidate_hover();
  void invalidate_selection();
  void get_selection_info(SelectionInfo *selectionInfo);

  void drag_selection(float dx, float dy, const Matrix4x4& worldTo3DH);

  MeshView *get_mesh_view();

  BSDF *get_bsdf();
  StaticScene::SceneObject *get_static_object();

  // MeshView methods
  void collapse_selected_edge();
  void flip_selected_edge();
  void split_selected_edge();
  void upsample();
  void downsample();
  void resample();

 private:

  // Helpers for render_in_opengl.
  void draw_faces() const;
  void draw_edges() const;
  void draw_feature_if_needed(const MeshFeature *feature) const;
  void draw_vertex(const Vertex *v) const;
  void draw_halfedge_arrow(const Halfedge *h) const;
  DrawStyle *get_draw_style(const HalfedgeElement *element) const;

  /**
   * Returns w for collision, and writes barycentric coordinates to baryPtr.
   */
  double triangle_selection_test_4d(const Vector2D& p, const Vector4D& A,
                                    const Vector4D& B, const Vector4D& C,
                                    Vector3D *baryPtr);

  /**
   * Returns t/f for collision, and writes barycentric coordinates to baryPtr.
   */
  bool triangle_selection_test_2d(const Vector2D& p, const Vector2D& A,
                                  const Vector2D& B, const Vector2D& C,
                                  float *uPtr, float *vPtr);

  /**
   * Given that hoveredFeature is pointing to a face, determines which
	 * subfeature (vertex, edge, halfedge, face) it's pointing to within
	 * that face.
   */
  void choose_hovered_subfeature();

  // selection draw styles
  MeshFeature potentialFeature, hoveredFeature, selectedFeature;
	DrawStyle *defaultStyle, *hoveredStyle, *selectedStyle;

  // halfEdge mesh
  HalfedgeMesh mesh;
  MeshResampler resampler;

  // material
  BSDF* bsdf;
};

} // namespace DynamicScene
} // namespace CMU462

#endif // CMU462_DYNAMICSCENE_MESH_H
