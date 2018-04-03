var gulp = require('gulp');
var plugins = require('gulp-load-plugins')();
var pump = require('pump');

gulp.task('compress', function () {
  pump([
    gulp.src('src/js/*.js'),
    plugins.uglify(),
    gulp.dest('./dist/')
  ]);
});

gulp.task('html', function () {
  gulp.src(['./src/**/index.jade'])
    .pipe(plugins.jade({ pretty: true, doctype: 'html' }))
    .on('error', plugins.util.log)
    .pipe(gulp.dest('.'));
});

gulp.task('css', function () {
  gulp.src(['./src/**/style.scss'])
    .pipe(plugins.sass({ outputStyle: 'compressed' })
      .on('error', plugins.sass.logError))
    .pipe(gulp.dest('./dist/'));
});

gulp.task('watch', function () {
  gulp.watch(['./src/**/*.jade'], ['html']);
  gulp.watch(['./src/**/*.scss'], ['css']);
});

gulp.task('default', ['compress', 'html', 'css', 'watch'], function () { });