#ifndef CMU462_DYNAMICSCENE_SCENE_H
#define CMU462_DYNAMICSCENE_SCENE_H

#include <string>
#include <vector>
#include <iostream>

#include "CMU462/CMU462.h"
#include "CMU462/color.h"

#include "GL/glew.h"

#include "draw_style.h"
#include "mesh_view.h"

#include "../bbox.h"
#include "../ray.h"
#include "../static_scene/scene.h"

namespace CMU462 { namespace DynamicScene {

struct SelectionInfo {
  std::vector<std::string> info;
};

/**
 * Interface that all physical objects in the scene conform to.
 * Note that this doesn't include properties like material that may be treated
 * as separate entities in a COLLADA file, or lights, which are treated
 * specially.
 */
class SceneObject {
 public:

  /**
   * Passes in logic for how to render the object in openGL.
   */
  virtual void set_draw_styles(DrawStyle *defaultStyle, DrawStyle *hoveredStyle,
                               DrawStyle *selectedStyle) = 0;
  /**
   * Renders the object in openGl, assuming that the camera and projection
   * matrices have already been set up.
   */
  virtual void render_in_opengl() const = 0;

  /**
   * Given a transformation matrix from local to space to world space, returns
   * a bounding box of the object in world space. Note that this doesn't have
   * to be the smallest possible bbox, in case that's difficult to compute.
   */
  virtual BBox get_bbox() = 0;

  /**
   * Finds the closest collision with this mesh and p in
   * unit-cube-space:
   * - If the w coordinate of this collision is smaller than minW, return it.
   * - If there is no collision with a smaller w coordinate than minW, return
   *   -1.
   * p: (x, y) in [(-1, -1)-(1, 1)] space.
   * modelTo3DH: converts model coordinates for this mesh into homogenous unit
   *             cube coordinates.
   * minW: min w found up to this point.
   * If the closest collision is closer than specified by minW, mark the feature
   * as tentatively selected.
   */
  virtual double test_selection(const Vector2D& p, const Matrix4x4& modelTo3DH,
                                double minW) = 0;

  /**
   * Marks a tentatively hover collision as genuine. This is needed because
   * multiple scene objects may have had tentative collisions, and a winner
   * needs to be chosen.
   */
  virtual void confirm_hover() = 0;

  /**
   * If this operation has a hovered feature, convert it to a selected feature.
   * Otherwise the behavior is undefined.
   */
  virtual void confirm_select() = 0;
  
  /**
   * Sets this object as not having any hovered elements.
   */
  virtual void invalidate_hover() = 0;

  /**
   * Removes any tentative or confirmed selections for this object.
   */
  virtual void invalidate_selection() = 0;

  /**
   * Returns info about the current selection (null if this object doesn't have
   * a selection), for use in drawHUD.
   */
  virtual void get_selection_info(SelectionInfo *selectionInfo) = 0;

  /**
   * If this object holds the selected element, drag it by (dx, dy). If this
   * object doesn't hold the selected element, do nothing.
   * (dx, dy) are in (-1, -1)-to-(1, 1) coordinates; a dx of 1 corresponds to
   * a translation of screenW.
   * modelTo3DH is used to convert from model space to 3D homogenous
   * coordinates, which are then divided into the unit cube.
   */
  virtual void drag_selection(float dx, float dy,
                              const Matrix4x4& modelTo3DH) = 0;

  /**
   * Returns this object as a MeshView if such a conversion is possible,
   * otherwise returns nullptr.
   */
  virtual MeshView *get_mesh_view() = 0;

  /**
   * Converts this object to an immutable, raytracer-friendly form. Passes in a
   * local-space-to-world-space transformation matrix, because the raytracer
   * expects all the objects to be
   */
  virtual StaticScene::SceneObject *get_static_object() = 0;
};


/**
 * A light.
 */
class SceneLight {
 public:
  virtual StaticScene::SceneLight *get_static_light() const = 0;
};

/**
 * The scene that meshEdit generates and works with.
 */
class Scene {
 public:
  Scene(std::vector<SceneObject *> objects, std::vector<SceneLight *> lights) {
    this->objects = objects;
    this->lights = lights;
    this->selectionIdx = -1;
    this->hoverIdx = -1;
  }

  /**
   * Passes instructions to every object in the scene for how to render
   * themselves in openGL.
   */
  void set_draw_styles(DrawStyle *defaultStyle, DrawStyle *hoveredStyle,
                       DrawStyle *selectedStyle);
  /**
   * Renders the scene in openGL, assuming the camera and projection
   * transformations have been applied elsewhere.
   */
  void render_in_opengl();

  /**
   * Gets a bounding box for the entire scene in world space coordinates.
   * May not be the tightest possible.
   */
  BBox get_bbox();

  /**
   * Finds the object pointed to by the given (x, y) point.
   * x and y are from -1 to 1, NOT screenW to screenH.
   * Note that hoverIdx (and therefore has_hover) is automatically updated every
   * time this function is called.
   */
  void update_selection(const Vector2D& p, const Matrix4x4& worldTo3DH);

  /**
   * Returns true iff there is a hovered feature in the scene.
   */
  bool has_hover();

  /**
   * Returns true iff there is a selected feature in the scene.
   */
  bool has_selection();

  /**
   * If the cursor is hovering over something, mark it as selected.
   */
  void confirm_selection();

  /**
   * Invalidates any currently selected/hovered elements.
   */
  void invalidate_selection();

  /**
   * If there is current selection and it's draggable, translate its unit-cube
   * location by (dx, dy).
   */
  void drag_selection(float dx, float dy, const Matrix4x4& worldTo3DH);

  /**
   * Returns information about the given selection, or nullptr if there is none.
   * Note that this object is still owned by the Scene, so it is invalidated on
   * selection updates, scene changes, and scene destruction.
   */
  SelectionInfo *get_selection_info();

  void collapse_selected_edge();
  void flip_selected_edge();
  void split_selected_edge();

  void upsample_selected_mesh();
  void downsample_selected_mesh();
  void resample_selected_mesh();

  /**
   * Builds a static scene that's equivalent to the current scene and is easier
   * to use in raytracing, but doesn't allow modifications.
   */
  StaticScene::Scene *get_static_scene();

 private:
  SelectionInfo selectionInfo;
  std::vector<SceneObject *> objects;
  std::vector<SceneLight *> lights;
  int selectionIdx, hoverIdx;

  /**
   * If there is a selected object and it's a mesh, returns it as a MeshView.
   * Otherwise, returns nullptr.
   */
  MeshView *get_selection_as_mesh();

  /**
   * Gets the selected object from the scene, returning nullptr if no object is
   * selected.
   */
  SceneObject *get_selection();
};

} // namespace DynamicScene
} // namespace CMU462

#endif // CMU462_DYNAMICSCENE_DYNAMICSCENE_H
