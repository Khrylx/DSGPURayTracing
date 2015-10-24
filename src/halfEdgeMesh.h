/*
 * HalfedgeMesh.h
 *
 * Written By Keenan Crane for 15-462 Assignment 2.
 */

/**
 * A HalfedgeMesh is a data structure that makes it easy to iterate over (and
 * modify) a polygonal mesh.  The basic idea is that each edge of the mesh
 * gets associated with two "halfedges," one on either side, that point in
 * opposite directions.  These halfedges essentially serve as the "glue"
 * between different mesh elements (vertices, edges, and faces).  A half edge
 * mesh has the same basic flavor as a tree or linked list data structure:
 * each node has pointers that reference other nodes.  In particular, each
 * half edge points to:
 *
 *    -its root vertex,
 *    -its associated edge,
 *    -the face it sits on,
 *    -its "twin", i.e., the halfedge on the other side of the edge,
 *    -and the next halfedge in cyclic order around the face.
 *
 * Vertices, edges, and faces each point to just one of their incident
 * halfedges.  For instance, an edge will point arbitrarily to either
 * its "left" or "right" halfedge.  Each vertex will point to one of
 * many halfedges leaving that vertex.  Each face will point to one of
 * many halfedges going around that face.  The fact that these choices
 * are arbitrary does not at all affect the practical use of this data
 * structure: they merely provide a starting point for iterating over
 * the local region (e.g., walking around a face, or visiting the
 * neighbors of a vertex).  A practical example of iterating around a
 * face might look like:
 *
 *    HalfEdgeIter h = myFace->halfedge();
 *    do
 *    {
 *       // do something interesting with h
 *       h = h->next();
 *    }
 *    while( h != myFace->halfEdge() );
 *
 * At each iteration we walk to the "next" halfedge, until we return
 * to the original starting point.  A slightly more interesting
 * example is iterating around a vertex:
 *
 *    HalfEdgeIter h = myVertex->halfedge();
 *    do
 *    {
 *       // do something interesting with h
 *       h = h->twin()->next();
 *    }
 *    while( h != myVertex->halfedge() );
 *
 * (Can you draw a picture that explains this iteration?)  A very
 * different kind of iteration is when we want to iterate over, say,
 * *all* the edges of a mesh:
 *
 *    for( EdgeIter e = mesh.edges.begin(); e != mesh.edges.end(); e++ )
 *    {
 *       // do something interesting with e
 *    }
 *
 * A very important consequence of the halfedge representation is that
 * ---by design---it can only represent manifold, orientable triangle
 * meshes.  I.e., every point should have a neighborhood that looks disk-
 * like, and you should be able to assign to each polygon a normal
 * direction such that all these normals "point the same way" as you walk
 * around the surface.
 *
 * At a high level, that's all there is to know about the half edge
 * data structure.  But it's worth making a few comments about how this
 * particular implementation works---especially how things like boundaries
 * are handled.  First and foremost, the "pointers" used in this
 * implementation are actually STL iterators.  STL stands for the "standard
 * template library," and is a basic part of C++ that provides some very
 * convenient and powerful data structures and algorithms---if you've never
 * looked at STL before, now would be a great time to get familiar!  At
 * a high level, STL iterators behave a lot like pointers: they don't store
 * data, but rather reference some data that is allocated elsewhere.  And
 * the syntax is also very similar; for instance, if p is an iterator, then
 * *p yields the value referred to by p.  (As for the rest, Google is a
 * terrific resource! :-))
 *
 * Rather than accessing raw iterators, the HalfedgeMesh encapsulates these
 * pointers using methods like Halfedge::twin(), Halfedge::next(), etc.  The
 * reason for this encapsulation (as in most object-oriented programming)
 * is that it allows the user to make changes to the internal representation
 * later down the line.  For instance, if you know that the connectivity of
 * the mesh is never going to change, you might be able to improve performance
 * by (internally) replacing the linked lists with fixed-length arrays,
 * without breaking any code that might have been written using the abstract
 * interface.  (There are deeper reasons for this kind of encapsulation
 * when working with polygon meshes, but that's a story for another time!)
 *
 * Finally, some surfaces have "boundary loops," e.g., a pair of pants has
 * three boundaries: one at the waist, and two at the ankles.  These boundaries
 * are represented by special faces in our halfedge mesh---in fact, rather than
 * being stored in the usual list of faces (HalfedgeMesh::faces), they are
 * stored in a separae list of boundary loops (HalfedgeMesh::boundaries).  Each
 * face (boundary or regular) also stored a flag Face::_isBoundary that
 * indicates whether or not it is a boundary.  This value can be queried via the
 * public method Face::isBoundary() (again: encapsulation!)  So for instance, if
 * I wanted to know the area of all polygons that touch a given vertex, I might
 * write some code like this:
 *
 *    double totalArea = 0.;
 *    HalfEdgeIter h = myVertex->halfedge();
 *    do
 *    {
 *       // don't add the area of boundary faces!
 *       if( !h->face()->isBoundary() )
 *       {
 *          totalArea != h->face()->area();
 *       }
 *       h = h->twin()->next();
 *    }
 *    while( h != myVertex->halfedge() );
 *
 * In other words, whenever I'm processing a face, I should stop and ask: is
 * this really a geometric face in my mesh?  Or is it just a "virtual" face
 * that represents a boundary loop?  Finally, for convenience, the halfedge
 * associated with a boundary vertex is the first halfedge on the boundary.
 * In other words, if we want to iterate over, say, all faces touching a
 * boundary vertex, we could write
 *
 *    HalfEdgeIter h = myBoundaryVertex->halfedge();
 *    do
 *    {
 *       // do something interesting with h
 *       h = h->twin()->next();
 *    }
 *    while( !h->isBoundary() );
 *
 * (Notice that this loop will never terminate for an interior vertex!)
 *
 * More documentation can be found in the inline comments below.
 */

#ifndef CMU462_HALFEDGEMESH_H
#define CMU462_HALFEDGEMESH_H

#include <set>
#include <map>
#include <list>
#include <vector>
#include <utility>
#include <iostream>

#include "CMU462/CMU462.h"  // Standard 462 Vectors, etc.

#include "collada/polymesh_info.h"

using namespace std;
using namespace CMU462;

// For code clarity, we often want to distinguish between
// an integer that encodes an index (an "ordinal" number)
// from an integer that encodes a size (a "cardinal" number).
typedef size_t Index;
typedef size_t Size;

namespace CMU462 {
/*
 * A HalfedgeMesh is comprised of four atomic element types:
 * vertices, edges, faces, and halfedges.
 */
class Vertex;
class Edge;
class Face;
class Halfedge;

/*
 * Rather than using raw pointers to mesh elements, we store references
 * as STL::iterators---for convenience, we give shorter names to these
 * iterators (e.g., EdgeIter instead of list<Edge>::iterator).
 */
typedef list<Vertex>::iterator VertexIter;
typedef list<Edge>::iterator EdgeIter;
typedef list<Face>::iterator FaceIter;
typedef list<Halfedge>::iterator HalfedgeIter;

/*
 * We also need "const" iterator types, for situations where a method takes
 * a constant reference or pointer to a HalfedgeMesh.  Since these types are
 * used so frequently, we will use "CIter" as a shorthand abbreviation for
 * "constant iterator."
 */
typedef list<Vertex>::const_iterator VertexCIter;
typedef list<Edge>::const_iterator EdgeCIter;
typedef list<Face>::const_iterator FaceCIter;
typedef list<Halfedge>::const_iterator HalfedgeCIter;

/*
 * Some algorithms need to know how to compare two iterators (which comes
 * first?)
 * Here we just say that one iterator comes before another if the address of the
 * object it points to is smaller.  (You should not have to worry about this!)
 */
inline bool operator<(const HalfedgeIter& i, const HalfedgeIter& j) {
  return &*i < &*j;
}
inline bool operator<(const VertexIter& i, const VertexIter& j) {
  return &*i < &*j;
}
inline bool operator<(const EdgeIter& i, const EdgeIter& j) {
  return &*i < &*j;
}
inline bool operator<(const FaceIter& i, const FaceIter& j) {
  return &*i < &*j;
}

// We also need to know how to compare const iterators.
inline bool operator<(const HalfedgeCIter& i, const HalfedgeCIter& j) {
  return &*i < &*j;
}
inline bool operator<(const VertexCIter& i, const VertexCIter& j) {
  return &*i < &*j;
}
inline bool operator<(const EdgeCIter& i, const EdgeCIter& j) {
  return &*i < &*j;
}
inline bool operator<(const FaceCIter& i, const FaceCIter& j) {
  return &*i < &*j;
}

/**
 * The elementAddress() function is defined only for convenience (and
 * readability), and returns the actual memory address associated with
 * a mesh element referred to by the given iterator.  (This is especially
 * helpful for things like debugging, where we want to check that one
 * element is properly pointing to another.)
 */
inline Halfedge* elementAddress(HalfedgeIter h) { return &(*h); }
inline Vertex* elementAddress(VertexIter v) { return &(*v); }
inline Edge* elementAddress(EdgeIter e) { return &(*e); }
inline Face* elementAddress(FaceIter f) { return &(*f); }

/**
 * Same thing, just for constant references.
 */
inline Halfedge const* elementAddress(HalfedgeCIter h) { return &(*h); }
inline Vertex const* elementAddress(VertexCIter v) { return &(*v); }
inline Edge const* elementAddress(EdgeCIter e) { return &(*e); }
inline Face const* elementAddress(FaceCIter f) { return &(*f); }

class EdgeRecord {
 public:
  EdgeRecord(void) {}
  EdgeRecord(EdgeIter& _edge);

  EdgeIter edge;
  Vector3D optimalPoint;
  double score;
};

inline bool operator<(const EdgeRecord& r1, const EdgeRecord& r2) {
  if (r1.score != r2.score) {
    return (r1.score < r2.score);
  }

  EdgeIter e1 = r1.edge;
  EdgeIter e2 = r2.edge;
  return &*e1 < &*e2;
}

/**
 * HalfedgeElement is the base type for all mesh elements (halfedges,
 * vertices, edges, and faces).  This type is used whenever we want
 * a pointer to a generic element (i.e., we don't know if it's a vertex
 * edge, face, or halfedge).  It is mainly used for debugging and
 * visualization, and should probably not be used for most actual mesh
 * processing tasks.
 */
class HalfedgeElement {
 public:
  /**
   * Check if the element is a halfEdge.
   * \return pointer to the half edge structure if the element is a half edge,
   * NULL otherwise.
   */
  Halfedge* getHalfedge(void);

  /**
   * Check if the element is a vertex.
   * \return pointer to the vertex structure if the element is a half edge, NULL
   * otherwise.
   */
  Vertex* getVertex(void);

  /**
   * Check if the element is an edge.
   * \return pointer to the edge structure if the element is an edge, NULL
   * otherwise.
   */
  Edge* getEdge(void);

  /**
   * Check if the element is a face.
   * \return pointer to the face structure if the element is a face, NULL
   * otherwise.
   */
  Face* getFace(void);

  /**
   * Destructor.
   */
  virtual ~HalfedgeElement(void) {}
};

/**
 * A Halfedge is the basic "glue" between mesh elements, pointing to
 * its associated vertex, edge, and face, as will as its twin and next
 * halfedges.
 */
class Halfedge : public HalfedgeElement {
 public:
  HalfedgeIter& twin(void) { return _twin; }  ///< access the twin half edge
  HalfedgeIter& next(void) { return _next; }  ///< access the next half edge
  VertexIter& vertex(void) {
    return _vertex;
  }  ///< access the vertex in the half edge
  EdgeIter& edge(void) {
    return _edge;
  }  ///< access the edge the half edge is on
  FaceIter& face(void) {
    return _face;
  }  ///< access the face the half edge is on

  HalfedgeCIter twin(void) const {
    return _twin;
  }  ///< access the twin half edge (const iterator)
  HalfedgeCIter next(void) const {
    return _next;
  }  ///< access the next half edge (comst iterator)
  VertexCIter vertex(void) const {
    return _vertex;
  }  ///< access the vertex in the half edge (const iterator)
  EdgeCIter edge(void) const {
    return _edge;
  }  ///< access the edge the half edge is on (const iterator)
  FaceCIter face(void) const {
    return _face;
  }  ///< access the face the half edge is on (const iterator)

  /**
   * Check if the edge is a boundary edge.
   * \return true if yes, false otherwise
   */
  bool isBoundary(void);

  /**
   * For convenience, this method sets all of the
   * neighbors of this halfedge to the given values.
   */
  void setNeighbors(HalfedgeIter next, HalfedgeIter twin, VertexIter vertex,
                    EdgeIter edge, FaceIter face) {
    _next = next;
    _twin = twin;
    _vertex = vertex;
    _edge = edge;
    _face = face;
  }

 protected:
  HalfedgeIter _twin;  ///< halfedge on the "other side" of the edge
  HalfedgeIter _next;  ///< next halfedge around the current face
  VertexIter _vertex;  ///< vertex at the "base" or "root" of this halfedge
  EdgeIter _edge;      ///< associated edge
  FaceIter _face;      ///< face containing this halfedge
};

/**
 * A Face is a single polygon in the mesh.
 */
class Face : public HalfedgeElement {
 public:
  /**
   * initializes the face, possibly setting its boundary flag
   * (by default, a Face does not encode a boundary loop)
   */
  Face(bool isBoundary = false) : _isBoundary(isBoundary) {}

  /**
   * Returns a reference to some halfedge of this face
   */
  HalfedgeIter& halfedge(void) { return _halfedge; }

  /**
   * Returns some halfedge of this face
   */
  HalfedgeCIter halfedge(void) const { return _halfedge; }

  /**
   * returns the number of edges (or equivalently, vertices) of this face
   */
  Size degree(void) const {
    Size d = 0;  // degree

    // walk around the face
    HalfedgeIter h = _halfedge;
    do {
      d++;  // increment the degree
      h = h->next();
    } while (h != _halfedge);  // done walking around the face

    return d;
  }

  /**
   * check if this face represents a boundary loop
   * \returns true if and only if this face represents a boundary loop, false
   * otherwise
   */
  bool isBoundary(void) { return _isBoundary; }

  /**
   * Get a unit face normal (computed via the area vector).
   * \returns a unit face normal (computed via the area vector).
   */
  Vector3D normal(void) const;

  Matrix4x4 quadric;

 protected:
  HalfedgeIter _halfedge;  ///< one of the halfedges of this face
  bool _isBoundary;        ///< boundary flag
};

/**
 * A Vertex encodes one of the mesh vertices
 */
class Vertex : public HalfedgeElement {
 public:
  /**
   * returns some halfedge rooted at this vertex (reference)
   */
  HalfedgeIter& halfedge(void) { return _halfedge; }

  /**
   * returns some halfedge rooted at this vertex
   */
  HalfedgeCIter halfedge(void) const { return _halfedge; }

  Vector3D position;  ///< vertex position

  /**
   * For Loop subdivision, this will be the updated position of the vertex
   */
  Vector3D newPosition;

  /**
   * For Loop subdivision, this flag should be true if and only if this
   * vertex is a new vertex created by subdivision (i.e., if it corresponds
   * to a vertex of the original mesh)
   */
  bool isNew;

  /**
   * computes the average of the neighboring vertex positions and stores it in
   * Vertex::centroid
   */
  void computeCentroid(void) {
    centroid = Vector3D( 0., 0., 0. ); // centroid
    double d = 0.; // degree (i.e., number of neighbors)

    // Iterate over neighbors.
    HalfedgeCIter h = halfedge();
    do {
       // Add the contribution of the neighbor,
       // and increment the number of neighbors.
       centroid += h->next()->vertex()->position;
       d += 1.;

       h = h->twin()->next();
    } while( h != halfedge() );

    centroid /= d; // compute the average
  }

  /**
   * average of neighbor positions, storing the value computed by
   * Vertex::computeCentroid()
   */
  Vector3D centroid;

  /**
   * Compute vertex normal
   * Compute the approximate unit normal at this vertex and store it in
   * Vertex::normal. The normal is computed by taking the area-weighted
   * average of the normals of neighboring triangles, then normalizing.
   */
  void computeNormal(void) {
    normal = Vector3D( 0., 0., 0. );
    Vector3D pi = position;

    // Iterate over neighbors.
    HalfedgeCIter h = halfedge();
    if (isBoundary()) {
      do {
         Vector3D pj = h->next()->vertex()->position;
         Vector3D pk = h->next()->next()->vertex()->position;
         normal += cross( pj-pi, pk-pi );
         h = h->next()->twin();
      } while( h != halfedge() );      
    } else {
      do {
         Vector3D pj = h->next()->vertex()->position;
         Vector3D pk = h->next()->next()->vertex()->position;
         normal += cross( pj-pi, pk-pi );
         h = h->twin()->next();
      } while( h != halfedge() );
    }

    normal.normalize();
  }

  /**
   * Vertex normal
   */
  Vector3D normal;

  // TODO : add texcoord support
  // Complex texcoord;  ///< vertex texture coordinate

  /**
   * Check if if this vertex is on the boundary of the surface
   * \return true if and only if this vertex is on the boundary
   * of the surface, false otherwise
   */
  bool isBoundary(void) const {
    // iterate over the halfedges incident on this vertex
    HalfedgeIter h = _halfedge;
    do {
      // check if the current halfedge is on the boundary
      if (h->isBoundary()) {
        return true;
      }

      // move to the next halfedge around the vertex
      h = h->twin()->next();
    } while (h != _halfedge);  // done iterating over halfedges

    return false;
  }

  /**
   * returns the number of edges (or equivalently, polygons) touching this
   * vertex
   */
  Size degree(void) const {
    Size d = 0;  // degree

    // iterate over halfedges incident on this vertex
    HalfedgeIter h = _halfedge;
    do {
      // don't count boundary loops
      if (!h->face()->isBoundary()) {
        d++;  // increment degree
      }

      // move to the next halfedge around the vertex
      h = h->twin()->next();
    } while (h != _halfedge);  // done iterating over halfedges

    return d;
  }

  Matrix4x4 quadric;

 protected:

  /**
   * one of the halfedges "rooted" or "based" at this vertex
   */
  HalfedgeIter _halfedge;

};

class Edge : public HalfedgeElement {
 public:
  /**
   * returns one of the two halfedges of this vertex (reference)
   */
  HalfedgeIter& halfedge(void) { return _halfedge; }

  /**
   * returns one of the two halfedges of this vertex
   */
  HalfedgeCIter halfedge(void) const { return _halfedge; }

  bool isBoundary(void);

  double length(void) const {
    Vector3D p0 = halfedge()->vertex()->position;
    Vector3D p1 = halfedge()->twin()->vertex()->position;

    return (p1 - p0).norm();
  }

  /**
   * For Loop subdivision, this will be the position for the edge midpoint
   */
  Vector3D newPosition;

  /**
   * For Loop subdivision, this flag should be true if and only if this edge
   * is a new edge created by subdivision (i.e., if it cuts across a triangle
   * in the original mesh)
   */
  bool isNew;

  EdgeRecord record;

 protected:

  /**
   * One of the two halfedges associated with this edge
   */
  HalfedgeIter _halfedge;

};

class HalfedgeMesh {
 public:
  /**
   * Constructor.
   */
  HalfedgeMesh(void) {}

  /**
   * The assignment operator does a "deep" copy of the halfedge mesh data
   * structure; in other words, it makes new instances of each mesh element,
   * and ensures that pointers in the copy point to the newly allocated elements
   * rather than elements in the original mesh. This behavior is especially
   * important for making assignments, since the mesh on the right-hand side of
   * an assignment may be temporary (hence any pointers to elements in this mesh
   * will become invalid as soon as it is released.)
   */
  const HalfedgeMesh& operator=(const HalfedgeMesh& mesh);

  /**
   * The copy constructor likewise does a "deep" copy of the mesh (via the
   * assignment operator).
   */
  HalfedgeMesh(const HalfedgeMesh& mesh);

  /**
   * This method initializes the halfedge data structure from a raw list of
   * polygons, where each input polygon is specified as a list of (0-based)
   * vertex indices. The input must describe a manifold, oriented surface,
   * where the orientation of a polygon is determined by the order of vertices
   * in the list.
   */
  void build(const vector<vector<Index> >& polygons,
             const vector<Vector3D>& vertexPositions);

  // These methods return the total number of elements of each type.
  Size nHalfedges(void) const {
    return halfedges.size();
  }  ///< get the number of halfedges
  Size nVertices(void) const {
    return vertices.size();
  }  ///< get the number of vertices
  Size nEdges(void) const { return edges.size(); }  ///< get the number of edges
  Size nFaces(void) const { return faces.size(); }  ///< get the number of faces
  Size nBoundaries(void) const {
    return boundaries.size();
  }  ///< get the number of boundaries

  /*
   * These methods return iterators to the beginning and end of the lists of
   * each type of mesh element.  For instance, to iterate over all vertices
   * one can write
   *
   *    for( VertexIter v = mesh.verticesBegin(); v != mesh.verticesEnd(); v++ )
   *    {
   *       // do something interesting with v
   *    }
   *
   * Note that we have both const and non-const versions of these functions;
   *when
   * a mesh is passed as a constant reference, we would instead write
   *
   *    for( VertexCIter v = ... )
   *
   * rather than VertexIter.
   */
  HalfedgeIter halfedgesBegin(void) { return halfedges.begin(); }
  HalfedgeCIter halfedgesBegin(void) const { return halfedges.begin(); }
  HalfedgeIter halfedgesEnd(void) { return halfedges.end(); }
  HalfedgeCIter halfedgesEnd(void) const { return halfedges.end(); }
  VertexIter verticesBegin(void) { return vertices.begin(); }
  VertexCIter verticesBegin(void) const { return vertices.begin(); }
  VertexIter verticesEnd(void) { return vertices.end(); }
  VertexCIter verticesEnd(void) const { return vertices.end(); }
  EdgeIter edgesBegin(void) { return edges.begin(); }
  EdgeCIter edgesBegin(void) const { return edges.begin(); }
  EdgeIter edgesEnd(void) { return edges.end(); }
  EdgeCIter edgesEnd(void) const { return edges.end(); }
  FaceIter facesBegin(void) { return faces.begin(); }
  FaceCIter facesBegin(void) const { return faces.begin(); }
  FaceIter facesEnd(void) { return faces.end(); }
  FaceCIter facesEnd(void) const { return faces.end(); }
  FaceIter boundariesBegin(void) { return boundaries.begin(); }
  FaceCIter boundariesBegin(void) const { return boundaries.begin(); }
  FaceIter boundariesEnd(void) { return boundaries.end(); }
  FaceCIter boundariesEnd(void) const { return boundaries.end(); }

  /*
   * These methods allocate new mesh elements, returning a pointer (i.e.,
   * iterator) to the new element.
   * (These methods cannot have const versions, because they modify the mesh!)
   */
  HalfedgeIter newHalfedge(void) {
    return halfedges.insert(halfedges.end(), Halfedge());
  }
  VertexIter newVertex(void) {
    return vertices.insert(vertices.end(), Vertex());
  }
  EdgeIter newEdge(void) { return edges.insert(edges.end(), Edge()); }
  FaceIter newFace(void) { return faces.insert(faces.end(), Face(false)); }
  FaceIter newBoundary(void) {
    return boundaries.insert(boundaries.end(), Face(true));
  }

  /*
   * These methods delete a specified mesh element.  One should think very, very
   * carefully about
   * exactly when and how to delete mesh elements, since other elements will
   * often still point
   * to the element that is being deleted, and accessing a deleted element will
   * cause your
   * program to crash (or worse!).  A good exercise to think about is: suppose
   * you're iterating
   * over a linked list, and want to delete some of the elements as you go.  How
   * do you do this
   * without causing any problems?  For instance, if you delete the current
   * element, will you be
   * able to iterate to the next element?  Etc.
   */
  void deleteHalfedge(HalfedgeIter h) { halfedges.erase(h); }
  void deleteVertex(VertexIter v) { vertices.erase(v); }
  void deleteEdge(EdgeIter e) { edges.erase(e); }
  void deleteFace(FaceIter f) { faces.erase(f); }
  void deleteBoundary(FaceIter b) { boundaries.erase(b); }

  /* For a triangle mesh, you will implement the following
   * basic edge operations.  (Can you generalize to other
   * polygonal meshes?)
   */

  /**
   * Flip an edge, returning a pointer to the flipped edge
   */
  EdgeIter flipEdge(EdgeIter e);

  /**
   * Split an edge, returning a pointer to the inserted midpoint vertex; the
   * halfedge of this vertex should refer to one of the edges in the original
   * mesh
   */
  VertexIter splitEdge(EdgeIter e);

  /**
   * collapse an edge, returning a pointer to the collapsed vertex
   */
  VertexIter collapseEdge(EdgeIter e);

 protected:

  /*
   * Here's where the mesh elements are actually stored---this is the one
   * and only place we have actual data (rather than pointers/iterators).
   */
  list<Halfedge> halfedges;
  list<Vertex> vertices;
  list<Edge> edges;
  list<Face> faces;
  list<Face> boundaries;

};  // class HalfedgeMesh

inline Halfedge* HalfedgeElement::getHalfedge(void) {
  return dynamic_cast<Halfedge*>(this);
}
inline Vertex* HalfedgeElement::getVertex(void) {
  return dynamic_cast<Vertex*>(this);
}
inline Edge* HalfedgeElement::getEdge(void) {
  return dynamic_cast<Edge*>(this);
}
inline Face* HalfedgeElement::getFace(void) {
  return dynamic_cast<Face*>(this);
}

}  // End of CMU 462 namespace.

#endif  // CMU462_HALFEDGEMESH_H
