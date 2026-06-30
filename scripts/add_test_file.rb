#!/usr/bin/env ruby
# Registers a test source file (already created on disk under WhisperScribeTests/)
# into the WhisperScribeTests target. Idempotent. Usage:
#   ruby scripts/add_test_file.rb WhisperScribeTests/WhisperModelTests.swift
require 'xcodeproj'

rel = ARGV[0] or abort 'usage: add_test_file.rb WhisperScribeTests/<File>.swift'
abort "file not found: #{rel}" unless File.exist?(rel)

PROJECT = 'WhisperScribe.xcodeproj'
proj = Xcodeproj::Project.open(PROJECT)
test = proj.targets.find { |t| t.name == 'WhisperScribeTests' } or abort 'test target not found'
group = proj.main_group['WhisperScribeTests'] || proj.main_group.new_group('WhisperScribeTests', 'WhisperScribeTests')

abs = File.expand_path(rel)
ref = proj.files.find { |f| f.real_path.to_s == abs } || group.new_file(File.basename(rel))
test.add_file_references([ref]) unless test.source_build_phase.files_references.include?(ref)

proj.save
puts "registered #{rel}"
