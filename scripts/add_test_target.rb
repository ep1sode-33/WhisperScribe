#!/usr/bin/env ruby
# Adds a host-based Swift Testing unit-test bundle "WhisperScribeTests" and
# wires it into the shared "WhisperScribe" scheme. Idempotent.
require 'xcodeproj'

PROJECT = 'WhisperScribe.xcodeproj'
proj = Xcodeproj::Project.open(PROJECT)

app = proj.targets.find { |t| t.name == 'WhisperScribe' } or abort 'app target not found'

test = proj.targets.find { |t| t.name == 'WhisperScribeTests' }
unless test
  test = proj.new_target(:unit_test_bundle, 'WhisperScribeTests', :osx, '14.0')
end

test.build_configurations.each do |c|
  s = c.build_settings
  s['PRODUCT_BUNDLE_IDENTIFIER'] = 'com.william.WhisperScribeTests'
  s['PRODUCT_NAME']              = '$(TARGET_NAME)'
  s['SWIFT_VERSION']            = '5.0'
  s['MACOSX_DEPLOYMENT_TARGET'] = '14.0'
  s['GENERATE_INFOPLIST_FILE']  = 'YES'
  s['CODE_SIGN_IDENTITY']       = '-'
  s['CODE_SIGN_STYLE']          = 'Automatic'
  s['SWIFT_EMIT_LOC_STRINGS']   = 'NO'
  s['TEST_HOST']    = '$(BUILT_PRODUCTS_DIR)/WhisperScribe.app/Contents/MacOS/WhisperScribe'
  s['BUNDLE_LOADER'] = '$(TEST_HOST)'
end

# Depend on the app so it builds + hosts the tests.
test.add_dependency(app) unless test.dependencies.any? { |d| d.target == app }

# A plain (non-synchronized) group for test sources.
proj.main_group['WhisperScribeTests'] || proj.main_group.new_group('WhisperScribeTests', 'WhisperScribeTests')

proj.save

# Wire the test target into the shared scheme.
scheme_path = File.join(Xcodeproj::XCScheme.shared_data_dir(PROJECT).to_s, 'WhisperScribe.xcscheme')
scheme = Xcodeproj::XCScheme.new(scheme_path)
already = scheme.test_action.testables.any? { |t| t.buildable_references.first&.target_name == 'WhisperScribeTests' }
unless already
  scheme.test_action.add_testable(Xcodeproj::XCScheme::TestAction::TestableReference.new(test))
  scheme.save_as(PROJECT, 'WhisperScribe', true)
end

puts 'WhisperScribeTests target ready.'
