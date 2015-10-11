#include "collada.h"
#include "math.h"

#include <assert.h>
#include <map>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#define stat(s) // cerr << "[COLLADA Parser] " << s << endl;

using namespace std;

namespace CMU462 {
namespace Collada {

SceneInfo* ColladaParser::scene; // pointer to output scene description

Vector3D ColladaParser::up; // scene up direction
Matrix4x4 ColladaParser::transform; // current transformation
map<string, XMLElement*> ColladaParser::sources; // URI lookup table

// Parser Helpers //

inline Color rgb_from_string ( string color_string ) {

  Color c;

  stringstream ss (color_string);
  ss >> c.r;
  ss >> c.g;
  ss >> c.b;
  c.a = 1.0;

  return c;

}

inline Color rgba_from_string ( string color_string ) {

  Color c;

  stringstream ss (color_string);
  ss >> c.r;
  ss >> c.g;
  ss >> c.b;
  ss >> c.a;

  return c;

}

void ColladaParser::uri_load( XMLElement* xml ) {

  if (xml->Attribute("id")) {
    string id = xml->Attribute("id");
    sources[id] = xml;
  }

  XMLElement* child = xml->FirstChildElement();
  while (child) {
    uri_load(child);
    child = child->NextSiblingElement();
  }

}

XMLElement* ColladaParser::uri_find( string id ) {

  return (sources.find(id) != sources.end()) ? sources[id] : NULL;

}


XMLElement* ColladaParser::get_element( XMLElement* xml, string query ) {

  stringstream ss (query);

  // find xml element
  XMLElement* e = xml; string token;
  while( e && getline(ss, token, '/') ) {
    e = e->FirstChildElement( token.c_str() );
  }

  // handle indirection
  if (e) {
		const char* url = e->Attribute("url");
    if (url) {
      string id = url + 1;
      e = uri_find(id);
    }
  }

  return e;
}


XMLElement* ColladaParser::get_technique_common( XMLElement* xml ) {

  XMLElement* common_profile = xml->FirstChildElement("profile_COMMON");
  if (common_profile) {
    XMLElement* technique = common_profile->FirstChildElement("technique");
    while (technique) {
      string sid = technique->Attribute("sid");
      if (sid == "common") return technique;
      technique = technique->NextSiblingElement("technique");
    }
  }

  return xml->FirstChildElement("technique_common");

}


XMLElement* ColladaParser::get_technique_462( XMLElement* xml ) {

  XMLElement* technique = xml->FirstChildElement("technique");
	while (technique) {
		string profile = technique->Attribute("profile");
		if (profile == "CMU462") return technique;
		technique = technique->NextSiblingElement("technique");
	}

	return NULL;

}


int ColladaParser::load( const char* filename, SceneInfo* sceneInfo ) {

  ifstream in (filename);
  if (!in.is_open()) {
    return -1;
  } in.close();

  XMLDocument doc;
  doc.LoadFile(filename);
  if (doc.Error()) {
    stat("XML error: ");
    doc.PrintError();
    exit(EXIT_FAILURE);
  }

  // Check XML schema
  XMLElement* root = doc.FirstChildElement("COLLADA");
  if (!root) {
    stat("Error: not a COLLADA file!")
    exit(EXIT_FAILURE);
  } else {
    stat("Loading COLLADA file...");
  }

	// Set output scene pointer
	scene = sceneInfo;

  // Build uri table
  uri_load(root);

  // Load assets - correct up direction
  if (XMLElement* e_asset = get_element(root, "asset")) {
    XMLElement* up_axis = get_element(e_asset, "up_axis");
		if (!up_axis) {
			stat("Error: No up direciton defined in COLLADA file");
			exit(EXIT_FAILURE);
		}

		// get up direction and correct non-Y_UP scenes by
		// setting a non-identity global entry transformation
		string up_dir = up_axis->GetText();
		transform = Matrix4x4::identity();
		if (up_dir == "X_UP") {

			// the swap X-Y matrix
			transform(0,0) =  0; transform(0,1) = 1;
			transform(1,0) = -1; transform(1,1) = 0;

			// local up direction for lights and cameras
			up = Vector3D(1,0,0);

		} else if (up_dir == "Z_UP") {

			// the swap Z-Y matrix
			transform(1,1) =  0; transform(1,2) = 1;
			transform(2,1) = -1; transform(2,2) = 0;

			// local up direction for lights and cameras
			up = Vector3D(0,0,1);

		} else if (up_dir == "Y_UP") {
			up = Vector3D(0,1,0); // no need to correct Y_UP as its used internally
		} else {
			stat("Erro: invalid up direction in COLLADA file");
			exit(EXIT_FAILURE);
		}
  }

  // Load scene -
  // A scene should only have one visual_scene instance, this constraint
  // creates a one-to-one relationship between the document, the top-level
  // scene, and its visual description (COLLADA spec 1.4 page 91)
  if (XMLElement* e_scene = get_element(root, "scene/instance_visual_scene")) {

    stat("Loading scene...");

    // parse all nodes in scene
		XMLElement* e_node = get_element(e_scene, "node");
	  while (e_node) {
			parse_node(e_node);
			e_node = e_node->NextSiblingElement("node");
		}

  } else {
    stat("Error: No scene description found in file:" << filename);
    return -1;
  }

  return 0;

}

int ColladaParser::save( const char* filename, const SceneInfo* sceneInfo ) {

  // TODO: not yet supported
  return 0;

}

void ColladaParser::parse_node( XMLElement* xml ) {

	// create new node
	Node node = Node();

  // name & id
  node.id   = xml->Attribute( "id" );
  node.name = xml->Attribute("name");
  stat(" |- Node: " << node.name << " (id:" << node.id << ")");

	// node transformation -
  // combine in order of declaration if the transformations are given as a
  // transformation list
  XMLElement* e = xml->FirstChildElement();
  while (e) {

    string name = e->Name();

    // transformation - matrix
    if (name == "matrix") {

      string s = e->GetText();
      stringstream ss (s);

      Matrix4x4 mat;
      ss >> mat(0,0); ss >> mat(0,1); ss >> mat(0,2); ss >> mat(0,3);
      ss >> mat(1,0); ss >> mat(1,1); ss >> mat(1,2); ss >> mat(1,3);
      ss >> mat(2,0); ss >> mat(2,1); ss >> mat(2,2); ss >> mat(2,3);
      ss >> mat(3,0); ss >> mat(3,1); ss >> mat(3,2); ss >> mat(3,3);

      node.transform = mat; break;

    }

    // transformation - rotate
    if (name == "rotate") {

      Matrix4x4 m;

      string s = e->GetText();
      stringstream ss (s);

      string sid = e->Attribute("sid");
      switch (sid.back()) {
        case 'X':
          ss >> m(1,1); ss >> m(1,2);
          ss >> m(2,1); ss >> m(2,2);
          break;
        case 'Y':
          ss >> m(0,0); ss >> m(2,0);
          ss >> m(0,2); ss >> m(2,2);
          break;
        case 'Z':
          ss >> m(0,0); ss >> m(0,1);
          ss >> m(1,0); ss >> m(1,1);
          break;
        default:
          break;
      }

      node.transform = m * node.transform;

    }

    // transformation - translate
    if (name == "translate") {

      Matrix4x4 m;

      string s = e->GetText();
      stringstream ss (s);

      ss >> m(0,3); ss >> m(1,3); ss >> m(2,3);

      node.transform = m * node.transform;
    }

    // transformation - scale
    if (name == "scale") {

      Matrix4x4 m;

      string s = e->GetText();
      stringstream ss (s);

      ss >> m(0,0); ss >> m(1,1); ss >> m(1,1);

      node.transform = m * node.transform;
    }

    // transformation - skew
    // Note (sky): ignored for now

    // transformation - lookat
    // Note (sky): ignored for now

    e = e->NextSiblingElement();
  }

	// push transformations
	Matrix4x4 transform_save = transform;

	// combine transformations
	node.transform = transform * node.transform;
	transform = node.transform;

	// parse child nodes if node is a joint
	XMLElement* e_child = get_element(xml, "node");
	while (e_child) {
		parse_node(e_child);
		e_child = e_child->NextSiblingElement("node");
	}

	// pop transformations
	transform = transform_save;

  // node instance -
  // non-joint nodes must contain a scene object instance
  XMLElement* e_camera   = get_element(xml, "instance_camera");
  XMLElement* e_light    = get_element(xml, "instance_light");
	XMLElement* e_geometry = get_element(xml, "instance_geometry");

  if (e_camera) {
    CameraInfo* camera = new CameraInfo();
    parse_camera( e_camera, *camera );
    node.instance = camera;
  } else if (e_light) {
    LightInfo* light = new LightInfo();
    parse_light( e_light, *light );
    node.instance = light;
  } else if (e_geometry) {
		if (get_element(e_geometry, "mesh")) {

			// mesh geometry
			PolymeshInfo* polymesh = new PolymeshInfo();
			parse_polymesh(e_geometry, *polymesh);

			// mesh material
			XMLElement* e_instance_material = get_element(xml,
			"instance_geometry/bind_material/technique_common/instance_material");
			if (e_instance_material) {

				if (!e_instance_material->Attribute("target")) {
					stat("Error: no targe material in instance: " << e_instance_material);
					exit(EXIT_FAILURE);
				}

				string material_id = e_instance_material->Attribute("target") + 1;
				XMLElement* e_material = uri_find(material_id);
				if (!e_material) {
					stat("Error: invalid target material id : " << material_id);
					exit(EXIT_FAILURE);
				}

				MaterialInfo* material = new MaterialInfo();
				parse_material(e_material, *material);
				polymesh->material = material;
			}

			node.instance = polymesh;

		} else if (get_element(e_geometry, "extra")) {

			// sphere geometry
			SphereInfo* sphere = new SphereInfo();
			parse_sphere(e_geometry, *sphere);

			// sphere material
			XMLElement* e_instance_material = get_element(xml,
			"instance_geometry/bind_material/technique_common/instance_material");
			if (e_instance_material) {

				if (!e_instance_material->Attribute("target")) {
					stat("Error: no targe material in instance: " << e_instance_material);
					exit(EXIT_FAILURE);
				}

				string material_id = e_instance_material->Attribute("target") + 1;
				XMLElement* e_material = uri_find(material_id);
				if (!e_material) {
					stat("Error: invalid target material id : " << material_id);
					exit(EXIT_FAILURE);
				}

				MaterialInfo* material = new MaterialInfo();
				parse_material(e_material, *material);
				sphere->material = material;
			}

			node.instance = sphere;

		}
  }

	// add node to scene
	scene->nodes.push_back(node);
}

void ColladaParser::parse_camera( XMLElement* xml, CameraInfo& camera ) {

  // name & id
  camera.id   = xml->Attribute( "id" );
  camera.name = xml->Attribute("name");
  camera.type = Instance::CAMERA;

	// default look direction is down the up axis
	camera.up_dir   = up;
	camera.view_dir = Vector3D(0,0,-1);

  // NOTE (sky): only supports perspective for now
  XMLElement* e_perspective = get_element(xml, "optics/technique_common/perspective");
  if (e_perspective) {
    XMLElement* e_xfov  = e_perspective->FirstChildElement("xfov" );
    XMLElement* e_yfov  = e_perspective->FirstChildElement("yfov" );
    XMLElement* e_znear = e_perspective->FirstChildElement("znear");
    XMLElement* e_zfar  = e_perspective->FirstChildElement("zfar" );

    camera.hFov  = e_xfov  ? atof(e_xfov  -> GetText()) : 50.0f;
    camera.vFov  = e_yfov  ? atof(e_yfov  -> GetText()) : 35.0f;
    camera.nClip = e_znear ? atof(e_znear -> GetText()) : 0.001f;
    camera.fClip = e_zfar  ? atof(e_zfar  -> GetText()) : 1000.0f;

    if (!e_yfov ) { // if vfov is not given, compute from aspect ratio
      XMLElement* e_aspect_ratio = get_element(e_perspective, "aspect_ratio");
      if (e_aspect_ratio) {
        float aspect_ratio = atof(e_aspect_ratio->GetText());
        camera.vFov = 2 * degrees(atan(tan(radians(0.5 * camera.hFov)) / aspect_ratio));
      } else {
        stat("Error: incomplete perspective definition in: " << camera.id);
        exit(EXIT_FAILURE);
      }
    }
  } else {
    stat("Error: no perspective defined in camera: " << camera.id );
    exit(EXIT_FAILURE);
  }

  // print summary
  stat("  |- " << camera);
}

void ColladaParser::parse_light( XMLElement* xml, LightInfo& light ) {

  // name & id
  light.id   = xml->Attribute( "id" );
  light.name = xml->Attribute("name");
  light.type = Instance::LIGHT;

  // common profile
  XMLElement* e_common = get_technique_common(xml);
  if (!e_common) {
    stat("Error: Common profile not defined in light: " << light.id);
    exit(EXIT_FAILURE);
  }

  // light parameters
  XMLElement* e_light = e_common->FirstChildElement();
  if (e_light) {

    // type
    string type = e_light->Name();

    // shared parameters
    XMLElement* e_color = get_element(e_light, "color");
    if (e_color) {
      string color_string = e_color->GetText();
      light.color = rgb_from_string( color_string );
    } else {
      stat("Error: No color definition in light: " << light.id);
      exit(EXIT_FAILURE);
    }

    // type-specific parameters
    if (type == "ambient") {
      light.light_type = LightType::AMBIENT;
    } else if (type == "directional") {
      light.light_type = LightType::DIRECTIONAL;
			light.direction = -up;
    } else if (type == "point") {
      light.light_type = LightType::POINT;
      XMLElement* e_constant_att = get_element(e_light, "constant_attenuation");
      if (e_constant_att) light.constant_att = atof(e_constant_att->GetText());
      XMLElement* e_linear_att = get_element(e_light, "linear_attenuation");
      if (e_linear_att) light.constant_att = atof(e_linear_att->GetText());
      XMLElement* e_quadratic_att = get_element(e_light, "quadratic_attenuation");
      if (e_quadratic_att) light.constant_att = atof(e_quadratic_att->GetText());
    } else if (type == "spot") {
      light.light_type = LightType::SPOT;
			light.direction = -up;
			XMLElement* e_falloff_deg = e_light->FirstChildElement("falloff_angle");
      if (e_falloff_deg) light.constant_att = atof(e_falloff_deg->GetText());
      XMLElement* e_falloff_exp = e_light->FirstChildElement("falloff_exponent");
      if (e_falloff_exp) light.constant_att = atof(e_falloff_exp->GetText());
      XMLElement* e_constant_att = get_element(e_light, "constant_attenuation");
      if (e_constant_att) light.constant_att = atof(e_constant_att->GetText());
      XMLElement* e_linear_att = get_element(e_light, "linear_attenuation");
      if (e_linear_att) light.constant_att = atof(e_linear_att->GetText());
      XMLElement* e_quadratic_att = get_element(e_light, "quadratic_attenuation");
      if (e_quadratic_att) light.constant_att = atof(e_quadratic_att->GetText());
    } else {
      stat("Error: Light type " << type << " in light: " << light.id << "is not supported");
      exit(EXIT_FAILURE);
    }

  }

  // print summary
  stat("  |- " << light);
}

void ColladaParser::parse_sphere(XMLElement* xml, SphereInfo& sphere) {

	// name & id
  sphere.id   = xml->Attribute( "id" );
  sphere.name = xml->Attribute("name");
  sphere.type = Instance::SPHERE;

	XMLElement* e_extra = xml->FirstChildElement("extra");
  if (!e_extra) {
    stat("Error: no extra element data defined in geometry: " << sphere.id);
    exit(EXIT_FAILURE);
  }

	XMLElement* e_technique = get_technique_462(e_extra);
	if (!e_technique) {
		stat("Error: no 462 profile technique in geometry: " << sphere.id);
    exit(EXIT_FAILURE);
	}

	XMLElement* e_radius = get_element(e_technique, "sphere/radius");
	if (!e_radius) {
		stat("Error: invalid sphere definition in geometry: " << sphere.id);
    exit(EXIT_FAILURE);
	}

	sphere.radius = atof(e_radius->GetText());

	// print summary
  stat("  |- " << sphere);
}


void ColladaParser::parse_polymesh(XMLElement* xml, PolymeshInfo& polymesh) {

  // name & id
  polymesh.id   = xml->Attribute( "id" );
  polymesh.name = xml->Attribute("name");
  polymesh.type = Instance::POLYMESH;

  XMLElement* e_mesh = xml->FirstChildElement("mesh");
  if (!e_mesh) {
    stat("Error: no mesh data defined in geometry: " << polymesh.id);
    exit(EXIT_FAILURE);
  }

  // array sources
  map< string, vector<float> > arr_sources;
  XMLElement* e_source = e_mesh->FirstChildElement("source");
  while (e_source) {

    // source id
    string source_id = e_source->Attribute("id");

    // source float array - other formats not handled
    XMLElement* e_float_array = e_source->FirstChildElement("float_array");
    if (e_float_array) {

      float f; vector<float> floats;

      // load float array string
      string s = e_float_array->GetText();
      stringstream ss (s);

      // load float array
      size_t num_floats = e_float_array->IntAttribute("count");
      for (size_t i = 0; i < num_floats; ++i) {
        ss >> f; floats.push_back(f);
      }

      // add to array sources
      arr_sources[source_id] = floats;
    }

    // parse next source
    e_source = e_source->NextSiblingElement("source");
  }

  // vertices
  vector<Vector3D> vertices; string vertices_id;
  XMLElement* e_vertices = e_mesh->FirstChildElement("vertices");
  if (!e_vertices) {
    stat("Error: no vertices defined in geometry: " << polymesh.id);
    exit(EXIT_FAILURE);
  } else {
    vertices_id = e_vertices->Attribute("id");
  }

  XMLElement* e_input = e_vertices->FirstChildElement("input");
  while (e_input) {

    // input semantic
    string semantic = e_input->Attribute("semantic");

    // semantic - position
    if (semantic == "POSITION") {
      string source = e_input->Attribute("source") + 1;
      if (arr_sources.find(source) != arr_sources.end()) {
        vector<float>& floats = arr_sources[source];
        size_t num_floats = floats.size();
        for (size_t i = 0; i < num_floats; i += 3) {
          Vector3D v = Vector3D(floats[i], floats[i+1], floats[i+2]);
          vertices.push_back(v);
        }
      } else {
        stat("Error: undefined input source: " << source);
        exit(EXIT_FAILURE);
      }
    }

    // NOTE (sky) : only positions are handled currently

    e_input = e_input->NextSiblingElement("input");
  }

  // polylist
  XMLElement* e_polylist = e_mesh->FirstChildElement("polylist");
  if (e_polylist) {

    // input arrays & array offsets
    bool has_vertex_array   = false; size_t vertex_offset   = 0;
    bool has_normal_array   = false; size_t normal_offset   = 0;
    bool has_texcoord_array = false; size_t texcoord_offset = 0;

    // input arr_sources
    XMLElement* e_input = e_polylist->FirstChildElement("input");
    while (e_input) {

      string semantic = e_input->Attribute("semantic");
      string source   = e_input->Attribute("source") + 1;
      size_t offset   = e_input->IntAttribute("offset");

      // vertex array source
      if (semantic == "VERTEX") {

        has_vertex_array = true;
        vertex_offset = offset;

        if (source == vertices_id) {
          polymesh.vertices.resize(vertices.size());
          copy(vertices.begin(), vertices.end(), polymesh.vertices.begin());
        } else {
          stat("Error: undefined source for VERTEX semantic: " << source);
          exit(EXIT_FAILURE);
        }
      }

      // normal array source
      if (semantic == "NORMAL") {

        has_normal_array = true;
        normal_offset = offset;

        if (arr_sources.find(source) != arr_sources.end()) {
          vector<float>& floats = arr_sources[source];
          size_t num_floats = floats.size();
          for (size_t i = 0; i < num_floats; i += 3) {
            Vector3D n = Vector3D(floats[i], floats[i+1], floats[i+2]);
            polymesh.normals.push_back(n);
          }
        } else {
          stat("Error: undefined source for NORMAL semantic: " << source);
          exit(EXIT_FAILURE);
        }
      }

      // texcoord array source
      if (semantic == "TEXCOORD") {

        has_texcoord_array = true;
        texcoord_offset = offset;

        if (arr_sources.find(source) != arr_sources.end()) {
          vector<float>& floats = arr_sources[source];
          size_t num_floats = floats.size();
          for (size_t i = 0; i < num_floats; i += 2) {
            Vector2D n = Vector2D(floats[i], floats[i+1]);
            polymesh.texcoords.push_back(n);
          }
        } else {
          stat("Error: undefined source for TEXCOORD semantic: " << source);
          exit(EXIT_FAILURE);
        }
      }

      e_input = e_input->NextSiblingElement("input");
    }

    // polygon info
    size_t num_polygons = e_polylist->IntAttribute("count");
    size_t stride = ( has_vertex_array   ? 1 : 0 ) +
                    ( has_normal_array   ? 1 : 0 ) +
                    ( has_texcoord_array ? 1 : 0 ) ;

    // create polygon size array and compute size of index array
    vector<size_t> sizes; size_t num_indices = 0;
    XMLElement* e_vcount = e_polylist->FirstChildElement("vcount");
    if (e_vcount) {

      size_t size;
      string s = e_vcount->GetText();
      stringstream ss (s);

      for (size_t i = 0; i < num_polygons; ++i) {
        ss >> size;
        sizes.push_back(size);
        num_indices += size * stride;
      }

    } else {
      stat("Error: polygon sizes undefined in geometry: " << polymesh.id);
      exit(EXIT_FAILURE);
    }

    // index array
    vector<size_t> indices;
    XMLElement* e_p = e_polylist->FirstChildElement("p");
    if (e_p) {

      size_t index;
      string s = e_p->GetText();
      stringstream ss (s);

      for (size_t i = 0; i < num_indices; ++i) {
        ss >> index;
        indices.push_back(index);
      }

    } else {
      stat("Error: no index array defined in geometry: " << polymesh.id);
      exit(EXIT_FAILURE);
    }

    // create polygons
    polymesh.polygons.resize(num_polygons);

    // vertex array indices
    if (has_vertex_array) {
      size_t k = 0;
      for (size_t i = 0; i < num_polygons; ++i) {
        for (size_t j = 0; j < sizes[i]; ++j) {
          polymesh.polygons[i].vertex_indices.push_back(
            indices[k * stride + vertex_offset]
          );
          k++;
        }
      }
    }

    // normal array indices
    if (has_normal_array) {
      size_t k = 0;
      for (size_t i = 0; i < num_polygons; ++i) {
        for (size_t j = 0; j < sizes[i]; ++j) {
          polymesh.polygons[i].normal_indices.push_back(
            indices[k * stride + normal_offset]
          );
          k++;
        }
      }
    }

    // texcoord array indices
    if (has_normal_array) {
      size_t k = 0;
      for (size_t i = 0; i < num_polygons; ++i) {
        for (size_t j = 0; j < sizes[i]; ++j) {
          polymesh.polygons[i].texcoord_indices.push_back(
            indices[k * stride + texcoord_offset]
          );
          k++;
        }
      }
    }

  }

  // print summary
  stat("  |- " << polymesh);
}

void ColladaParser::parse_material ( XMLElement* xml, MaterialInfo& material ) {

  // name & id
  material.id   = xml->Attribute( "id" );
  material.name = xml->Attribute("name");
  material.type = Instance::MATERIAL;

  // parse effect
	XMLElement* e_effect = get_element(xml, "instance_effect");
  if (e_effect) {

    // common
    XMLElement* e_technique = get_technique_common(e_effect);
    if (!e_technique) {
      stat("Error: no technique defined for common profile in material: " << material.id);
      exit(EXIT_FAILURE);
    }

    // phong
    XMLElement* e_phong = e_technique->FirstChildElement("phong");
    if (!e_phong) {
      stat("Error: no phong shading is defined for material: " << material.id);
      exit(EXIT_FAILURE);
    }

    Color none = Color(0,0,0,0);

    XMLElement* e_Ce = get_element(e_phong, "emission/color");
    XMLElement* e_Ca = get_element(e_phong, "ambient/color");
    XMLElement* e_Cd = get_element(e_phong, "diffuse/color");
    XMLElement* e_Cs = get_element(e_phong, "specular/color");
    XMLElement* e_Ns = get_element(e_phong, "shininess/float");
    XMLElement* e_Ni = get_element(e_phong, "index_of_refraction/float");

    material.Ce = e_Ce ? rgba_from_string(string(e_Ce->GetText())) : none;
    material.Ca = e_Ca ? rgba_from_string(string(e_Ca->GetText())) : none;
    material.Cd = e_Cd ? rgba_from_string(string(e_Cd->GetText())) : none;
    material.Cs = e_Cs ? rgba_from_string(string(e_Cs->GetText())) : none;
    material.Ns = e_Ns ? atof(e_Ns->GetText()) : 0.0f;
    material.Ni = e_Ni ? atof(e_Ni->GetText()) : 1.0f;

  } else {
    stat("Error: target effects not found for material: " << material.id);
    exit(EXIT_FAILURE);
  }

  // print summary
  stat("  |- " << material);
}

} // namespace Collada
} // namespace CMU462
