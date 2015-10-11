#include "mesh.h"

#include <cassert>
#include <sstream>

#include "../static_scene/object.h"

using std::ostringstream;

namespace CMU462 { namespace DynamicScene {

// For use in choose_hovered_subfeature.
static const double low_threshold  = .1;
static const double mid_threshold  = .2;
static const double high_threshold = 1.0 - low_threshold;


Mesh::Mesh(Collada::PolymeshInfo& polyMesh, const Matrix4x4& transform) {

	// Build halfedge mesh from polygon soup
  vector< vector<size_t> > polygons;
  for (Collada::PolyListIter p = polyMesh.polygons.begin();
       p != polyMesh.polygons.end(); p++) {
     polygons.push_back(p->vertex_indices);
  }
  vector<Vector3D> vertices = polyMesh.vertices; // DELIBERATE COPY.
  for (int i = 0; i < vertices.size(); i++) {
    vertices[i] = (transform * Vector4D(vertices[i], 1)).projectTo3D();
  }

  mesh.build(polygons, vertices);

  if (polyMesh.material) {
    material = new Material(*polyMesh.material);
  } else {
    material = new Material();
  }
}


/**************
 * RENDER FNS *
 **************/

void Mesh::render_in_opengl() const {

	// Enable lighting for faces
  glEnable(GL_LIGHTING);
  draw_faces();

  // Edges are drawn with flat shading.
  glDisable(GL_LIGHTING);
  draw_edges();

  // ONLY draw a vertex/half-edge if it's selected.
  draw_feature_if_needed(&hoveredFeature);
  draw_feature_if_needed(&selectedFeature);
  glEnable(GL_LIGHTING);
}

void Mesh::draw_faces() const {

  material->set_material_properties();
  for (FaceCIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++) {

    // Prevent z fighting (faces bleeding into edges and points).
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 1.0);

    DrawStyle* style = get_draw_style(elementAddress(f));
    if (style != defaultStyle) {
      glDisable(GL_LIGHTING);
      style->style_face();
    }

    glBegin(GL_POLYGON);
    Vector3D normal(f->normal());
    glNormal3dv(&normal.x);
    HalfedgeCIter h = f->halfedge();
    do {
      glVertex3dv(&h->vertex()->position.x);
      h = h->next();
    } while (h != f->halfedge());
    glEnd();

    if (style != defaultStyle) {
      glEnable(GL_LIGHTING);
    }
  }
}

void Mesh::draw_edges() const {
  defaultStyle->style_edge();
  for(EdgeCIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
    DrawStyle *style = get_draw_style(elementAddress(e));
    if (style != defaultStyle) {
      style->style_edge();
    }

    glBegin(GL_LINES);
    glVertex3dv(&e->halfedge()->vertex()->position.x);
    glVertex3dv(&e->halfedge()->twin()->vertex()->position.x);
    glEnd();

    if (style != defaultStyle) {
      defaultStyle->style_edge();
    }
  }
}

void Mesh::draw_feature_if_needed(const MeshFeature *feature) const {
  if (!feature->isValid()) return;
  glDisable(GL_DEPTH_TEST);

  const Vertex *v = feature->element->getVertex();
  if (v != nullptr) {
    draw_vertex(v);
  }
  const Halfedge *h = feature->element->getHalfedge();
  if (h != nullptr) {
    draw_halfedge_arrow(h);
  }

  glEnable(GL_DEPTH_TEST);
}

void Mesh::draw_vertex(const Vertex *v) const {
  get_draw_style(v)->style_vertex();
  glBegin(GL_POINTS);
  glVertex3d(v->position.x, v->position.y, v->position.z);
  glEnd();
}

void Mesh::draw_halfedge_arrow(const Halfedge *h) const {
  get_draw_style(h)->style_halfedge();

  const Vector3D& p0 = h->vertex()->position;
  const Vector3D& p1 = h->next()->vertex()->position;
  const Vector3D& p2 = h->next()->next()->vertex()->position;

  const Vector3D& e01 = p1-p0;
  const Vector3D& e12 = p2-p1;
  const Vector3D& e20 = p0-p2;

  const Vector3D& u = (e01 - e20) / 2;
  const Vector3D& v = (e12 - e01) / 2;

  const Vector3D& a = p0 + u / 5;
  const Vector3D& b = p1 + v / 5;

  const Vector3D& s = (b-a) / 5;
  const Vector3D& t = cross(h->face()->normal(), s);
  double theta = PI * 5 / 6;
  const Vector3D& c = b + cos(theta) * s + sin(theta) * t;

  glBegin(GL_LINE_STRIP);
  glVertex3dv(&a.x);
  glVertex3dv(&b.x);
  glVertex3dv(&c.x);
  glEnd();
}

DrawStyle *Mesh::get_draw_style(const HalfedgeElement *element) const {
  if (element == selectedFeature.element) return selectedStyle;
  if (element == hoveredFeature.element) return hoveredStyle;
  return defaultStyle;
}

void Mesh::set_draw_styles(DrawStyle *defaultStyle, DrawStyle *hoveredStyle,
                           DrawStyle *selectedStyle) {
  this->defaultStyle = defaultStyle;
  this->hoveredStyle = hoveredStyle;
  this->selectedStyle = selectedStyle;
}


/************
 * MISC FNS *
 ************/

BBox Mesh::get_bbox() {
  BBox bbox;
  for (VertexIter it = mesh.verticesBegin(); it != mesh.verticesEnd(); it++) {
    bbox.expand(it->position);
  }
  return bbox;
}


/*****************
 * SELECTION FNS *
 *****************/

double Mesh::test_selection(const Vector2D& p,
                            const Matrix4x4& worldTo3DH, double minW) {
  for(FaceIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++) {
    // Transform the face vertices into homogenous coordinates, where the x, y,
    // and z are perspective-divided by w and w is left unchanged.
    Vector4D A(f->halfedge()->vertex()->position, 1);
    Vector4D B(f->halfedge()->next()->vertex()->position, 1);
    Vector4D C(f->halfedge()->next()->next()->vertex()->position, 1);

    A = worldTo3DH * A;
    double wA = A.w;
    A /= wA;
    A.w = wA;

    B = worldTo3DH * B;
    double wB = B.w;
    B /= wB;
    B.w = wB;

    C = worldTo3DH * C;
    double wC = C.w;
    C /= wC;
    C.w = wC;

    Vector3D barycentricCoordinates;
    double w = triangle_selection_test_4d(p, A, B, C,
                                          &barycentricCoordinates);
    if (w > 0 && (minW < 0 || w < minW)) {
      // Update the record of the closest feature seen so far; note that the value of w
      // was already updated in our call to triangleSelectionTest.
      potentialFeature.bCoords = barycentricCoordinates;
      potentialFeature.element = elementAddress(f);
      potentialFeature.w = w;
      minW = w;
    }
  }
  return minW;
}

void Mesh::confirm_hover() {
  hoveredFeature = potentialFeature;
  potentialFeature.invalidate();
  choose_hovered_subfeature();
}

void Mesh::confirm_select() {
  selectedFeature = hoveredFeature;
}

void Mesh::invalidate_selection() {
  potentialFeature.invalidate();
  hoveredFeature.invalidate();
  selectedFeature.invalidate();
}

SelectionInfo *Mesh::get_selection_info() {
  if (!selectedFeature.isValid()) {
    return nullptr;
  }
  SelectionInfo *selectionInfo = new SelectionInfo();

  Vertex *v = selectedFeature.element->getVertex();
  if (v != nullptr) {
    ostringstream m1, m2, m3, m4, m5, m6, m7, m8;
    m1 << "VERTEX DATA";
    m2 << "address      = " << v;

    // -- Nicely format position data.
    const Vector3D & pos = v->position;
    m3 << scientific;
    m3.precision(4);
    m3 << "position:  x = " << pos.x;
    m4 << scientific;
    m4.precision(4);
    m4 << "           y = " << pos.y;
    m5 << scientific;
    m5.precision(4);
    m5 << "           z = " << pos.z;

    m6 << "halfedge()   = " << elementAddress(v -> halfedge());
    m7 << "isBoundary() = " << v -> isBoundary();
    m8 << "degree()     = " << v->degree();
    selectionInfo->info.reserve(8);
    selectionInfo->info.push_back(m1.str());
    selectionInfo->info.push_back(m2.str());
    selectionInfo->info.push_back(m3.str());
    selectionInfo->info.push_back(m4.str());
    selectionInfo->info.push_back(m5.str());
    selectionInfo->info.push_back(m6.str());
    selectionInfo->info.push_back(m7.str());
    selectionInfo->info.push_back(m8.str());
    return selectionInfo;
  }

  Halfedge* h = selectedFeature.element->getHalfedge();
  if (h != nullptr) {
    ostringstream m1, m2, m3, m4, m5, m6, m7, m8;
    m1 << "HALFEDGE DATA";
    m2 << "address      = " << h;
    m3 << "twin()       = " << elementAddress(h->twin());
    m4 << "next()       = " << elementAddress(h->next());
    m5 << "vertex()     = " << elementAddress(h->vertex());
    m6 << "edge()       = " << elementAddress(h->edge());
    m7 << "face()       = " << elementAddress(h->face());
    m8 << "isBoundary() = " << h->isBoundary() << endl;
    selectionInfo->info.reserve(8);
    selectionInfo->info.push_back(m1.str());
    selectionInfo->info.push_back(m2.str());
    selectionInfo->info.push_back(m3.str());
    selectionInfo->info.push_back(m4.str());
    selectionInfo->info.push_back(m5.str());
    selectionInfo->info.push_back(m6.str());
    selectionInfo->info.push_back(m7.str());
    selectionInfo->info.push_back(m8.str());
    return selectionInfo;
  }

  Edge* e = selectedFeature.element->getEdge();
  if (e != nullptr) {
    ostringstream m1, m2, m3, m4;
    m1 << "EDGE DATA";
    m2 << "address      = " << e;
    m3 << "halfedge()   = " << elementAddress(e->halfedge());
    m4 << "isBoundary() = " << e->isBoundary() << endl;
    selectionInfo->info.reserve(4);
    selectionInfo->info.push_back(m1.str());
    selectionInfo->info.push_back(m2.str());
    selectionInfo->info.push_back(m3.str());
    selectionInfo->info.push_back(m4.str());
    return selectionInfo;
  }

  Face* f = selectedFeature.element->getFace();
  if (f != nullptr) {
    ostringstream m1, m2, m3, m4, m5;
    m1 << "FACE DATA";
    m2 << "address      = " << f << endl;
    m3 << "halfedge()   = " << elementAddress( f->halfedge() ) << endl;
    m4 << "degree()     = " << f->degree() << endl;
    m5 << "isBoundary() = " << f->isBoundary() << endl;
    selectionInfo->info.reserve(8);
    selectionInfo->info.push_back(m1.str());
    selectionInfo->info.push_back(m2.str());
    selectionInfo->info.push_back(m3.str());
    selectionInfo->info.push_back(m4.str());
    selectionInfo->info.push_back(m5.str());
    return selectionInfo;
  }

  assert(false);
}

void Mesh::drag_selection(float dx, float dy, const Matrix4x4& worldTo3DH) {
  // Get selection as a 4D vector.
  if (!selectedFeature.isValid()) {
    return;
  }
  Vertex *v = selectedFeature.element->getVertex();
  if (v == nullptr) {
    return;
  }
  Vector4D pos(v->position, 1.0);

  // Translate to unit cube.
  pos = worldTo3DH * pos;
  double w = pos.w;
  pos /= w;

  // Shift by (dx, dy).
  pos.x += dx;
  pos.y += dy;

  // Translate back to model space.
  pos *= w;
  pos = worldTo3DH.inv() * pos;

  v->position = pos.to3D();
}

void Mesh::collapse_selected_edge() {
  HalfedgeElement *element = selectedFeature.element;
  if (element == nullptr) return;
  Edge *edge = element->getEdge();
  if (edge == nullptr) return;
  mesh.collapseEdge(edge->halfedge()->edge());
  invalidate_selection();
}

void Mesh::flip_selected_edge() {
  HalfedgeElement *element = selectedFeature.element;
  if (element == nullptr) return;
  Edge *edge = element->getEdge();
  if (edge == nullptr) return;
  mesh.flipEdge(edge->halfedge()->edge());
  invalidate_selection();
}

void Mesh::split_selected_edge() {
  HalfedgeElement *element = selectedFeature.element;
  if (element == nullptr) return;
  Edge *edge = element->getEdge();
  if (edge == nullptr) return;
  mesh.splitEdge(edge->halfedge()->edge());
  invalidate_selection();
}

void Mesh::upsample() {
  resampler.upsample(mesh);
  invalidate_selection();
}

void Mesh::downsample() {
  resampler.downsample(mesh);
  invalidate_selection();
}

void Mesh::resample() {
  resampler.resample(mesh);
  invalidate_selection();
}


double Mesh::triangle_selection_test_4d(const Vector2D& p, const Vector4D& A,
                                        const Vector4D& B, const Vector4D& C,
                                        Vector3D *baryPtr) {
  Vector2D a2D(A.x, A.y);
  Vector2D b2D(B.x, B.y);
  Vector2D c2D(C.x, C.y);
  float bU, bV;

  if (triangle_selection_test_2d(p, a2D, b2D, c2D, &bU, &bV)) {
    baryPtr->x = 1.0 - bU - bV;
    baryPtr->y = bV;
    baryPtr->z = bU;
    return A.w + (C.w - A.w) * bU + (B.w - A.w) * bV;
  }
  return -1.0;
}


bool Mesh::triangle_selection_test_2d(const Vector2D& p, const Vector2D& A,
                                      const Vector2D& B, const Vector2D& C,
                                      float *uPtr, float *vPtr) {
  // Compute vectors
  Vector2D v0 = C - A;
  Vector2D v1 = B - A;
  Vector2D v2 = p - A;

  // Compute dot products
  double dot00 = dot(v0, v0);
  double dot01 = dot(v0, v1);
  double dot02 = dot(v0, v2);
  double dot11 = dot(v1, v1);
  double dot12 = dot(v1, v2);

  // Compute barycentric coordinates
  double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
  double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  // Check if point is in triangle.
  bool output = (u >= 0) && (v >= 0) && (u + v < 1);

  if (output) {
    *uPtr = u;
    *vPtr = v;
  }

  return output;
}


/*
 * Given an event on a face (position in barycentric coordinates), determines
 * what feature of the face was selected (face, edges, halfedges, vertices).
 * NOTE: if your mesh has non-triangles in it, this will perform poorly.
 */
void Mesh::choose_hovered_subfeature() {
  Face* f = hoveredFeature.element->getFace();
  if (f == nullptr) {
    cerr << "choose_hovered_subfeature called with a non-face selected" << endl;
    exit(1);
  }

  // Grab the three halfedges of the triangle under the cursor.
  HalfedgeIter h1 = f->halfedge();
  HalfedgeIter h2 = h1->next();
  HalfedgeIter h3 = h2->next();

  // Check if the cursor is closest to a vertex; if so, this is the feature we want to return.
  // Check vertices, then edges, then halfedges, then default to the face.
  if(hoveredFeature.bCoords.x > high_threshold) {
    hoveredFeature.element = elementAddress(h1->vertex());
  } else if (hoveredFeature.bCoords.y > high_threshold) {
    hoveredFeature.element = elementAddress(h2->vertex());
  } else if (hoveredFeature.bCoords.z > high_threshold) {
    hoveredFeature.element = elementAddress(h3->vertex());
  } else if (hoveredFeature.bCoords.z < low_threshold) {
    hoveredFeature.element = elementAddress(h1->edge());
  } else if (hoveredFeature.bCoords.x < low_threshold) {
    hoveredFeature.element = elementAddress(h2->edge());
  } else if (hoveredFeature.bCoords.y < low_threshold) {
    hoveredFeature.element = elementAddress(h3->edge());
  } else if(hoveredFeature.bCoords.z < mid_threshold) {
    hoveredFeature.element = elementAddress(h1);
  } else if(hoveredFeature.bCoords.x < mid_threshold) {
    hoveredFeature.element = elementAddress(h2);
  } else if(hoveredFeature.bCoords.y < mid_threshold) {
    hoveredFeature.element = elementAddress(h3);
  } else {
    hoveredFeature.element = f;
  }
}

MeshView *Mesh::get_mesh_view() {
  return this;
}

StaticScene::SceneObject *Mesh::get_static_object() {
  return new StaticScene::Mesh(mesh);
}


} // namespace DynamicScene
} // namespace CMU462
