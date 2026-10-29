import 'dart:convert';

/// Каталог не запускает CV и не принимает решение о подтверждении фотографии.
class PoiMappingException implements Exception {
  final String code;
  final String message;
  const PoiMappingException(this.code, this.message);
  @override
  String toString() => '$code: $message';
}

class PoiBinding {
  final int classIndex;
  final String cvPoiId;
  final String name;
  final String? dgisPoiId;
  final String? candidateDgisPoiId;
  final String bindingStatus;
  final String sdkValidationStatus;

  const PoiBinding(this.classIndex, this.cvPoiId, this.name, this.dgisPoiId,
      this.candidateDgisPoiId, this.bindingStatus, this.sdkValidationStatus);
}

class PoiCatalog {
  final String version;
  final Map<int, PoiBinding> _byIndex;
  final Map<String, PoiBinding> _byCv;
  final Map<String, PoiBinding> _byDgis;
  PoiCatalog._(this.version, this._byIndex, this._byCv, this._byDgis);

  /// modelMapping должен быть взят из того же выпуска, что мобильная модель.
  factory PoiCatalog.fromJson(
      String json, Map<String, int> modelMapping) {
    final data = jsonDecode(json) as Map<String, dynamic>;
    if (data['schema_version'] != 1 ||
        data['mapping_version'] is! String || modelMapping.isEmpty ||
        modelMapping.values.toSet().length != modelMapping.length ||
        !List.generate(modelMapping.length, (i) => i)
            .every(modelMapping.values.contains)) {
      throw const FormatException('Invalid catalog or model mapping');
    }
    final indices = <int, PoiBinding>{};
    final cv = <String, PoiBinding>{};
    final dgis = <String, PoiBinding>{};
    for (final raw in data['entries'] as List<dynamic>) {
      final row = raw as Map<String, dynamic>;
      final index = row['class_index'];
      final id = row['cv_poi_id'];
      if (index is! int || id is! String || modelMapping[id] != index ||
          indices.containsKey(index) || cv.containsKey(id)) {
        throw const FormatException('Model class mapping mismatch');
      }
      final active = row['dgis_poi_id'];
      final candidate = row['candidate_dgis_poi_id'];
      for (final value in [active, candidate]) {
        if (value != null &&
            (value is! String || !RegExp(r'^[0-9]+$').hasMatch(value))) {
          throw const FormatException('2GIS IDs must be exact digit strings');
        }
      }
      final status = row['binding_status'] as String;
      final sdk = row['sdk_validation_status'] as String;
      if (!['mapped', 'scope_mismatch', 'unmapped'].contains(status) ||
          !['pending', 'verified'].contains(sdk) ||
          (status == 'mapped') != (active != null)) {
        throw const FormatException('Invalid binding status');
      }
      final binding = PoiBinding(index, id, row['name'] as String,
          active as String?, candidate as String?, status, sdk);
      indices[index] = binding;
      cv[id] = binding;
      for (final external in [active, candidate].whereType<String>().toSet()) {
        if (dgis.containsKey(external)) {
          throw const FormatException('Ambiguous 2GIS ID');
        }
        dgis[external] = binding;
      }
    }
    if (cv.length != modelMapping.length) {
      throw const FormatException('Catalog does not cover model classes');
    }
    return PoiCatalog._(data['mapping_version'] as String, indices, cv, dgis);
  }

  PoiBinding bindingForClassIndex(int index) {
    final binding = _byIndex[index];
    if (binding == null) {
      throw const PoiMappingException('unknown_class_index', 'Unknown model output');
    }
    return binding;
  }

  String targetForDgis(String id, {bool allowUnverified = false}) {
    if (!RegExp(r'^[0-9]+$').hasMatch(id)) {
      throw const PoiMappingException('invalid_dgis_id', 'Expected an exact digit string');
    }
    final binding = _byDgis[id];
    if (binding == null) {
      throw const PoiMappingException('dgis_mapping_missing', 'No CV binding');
    }
    _requireBinding(binding, allowUnverified);
    return binding.cvPoiId;
  }

  String dgisForCv(String cvPoiId, {bool allowUnverified = false}) {
    final binding = _byCv[cvPoiId];
    if (binding == null) {
      throw const PoiMappingException('unknown_cv_poi', 'Unknown CV target');
    }
    _requireBinding(binding, allowUnverified);
    return binding.dgisPoiId!;
  }

  void _requireBinding(PoiBinding binding, bool allowUnverified) {
    if (binding.bindingStatus != 'mapped') {
      throw const PoiMappingException('dgis_mapping_blocked', 'Object scope mismatch');
    }
    if (binding.sdkValidationStatus != 'verified' && !allowUnverified) {
      throw const PoiMappingException('dgis_mapping_unverified', 'SDK validation pending');
    }
  }
}
