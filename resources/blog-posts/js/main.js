/**
 * Main JavaScript file for additional main site functionality.
 *
 * @author  Joeri Hermans
 * @version 0.1
 * @since   8 July 2016
 */

var rippleEffect = function(e) {
  var target = e.target;
  var rectangle = target.getBoundingClientRect();
  var ripple = target.querySelector('.ripple');
  if( !ripple ) {
    ripple = document.createElement('span');
    ripple.className = 'ripple';
    ripple.style.height = ripple.style.width = Math.max(rectangle.width, rectangle.height) + 'px';
    // Check if the target has a first child.
    if( target.firstChild )
      target.insertBefore(ripple, target.firstChild);
    else
      target.appendChild(ripple);
  }
  // Check if we need to add a red ripple.
  if( target.classList.contains('red') )
    ripple.classList.add('ripple-red');
  ripple.classList.remove('show');
  var top = e.pageY - rectangle.top - ripple.offsetHeight / 2 - document.body.scrollTop;
  var left = e.pageX - rectangle.left - ripple.offsetWidth / 2 - document.body.scrollLeft;
  ripple.style.top = top + 'px';
  ripple.style.left = left + 'px';
  ripple.classList.add('show');

  return false;
};

function addRippleEffects() {
  // Add the ripple effect to all buttons in the page.
  var elements = document.getElementsByClassName("ripple-button");
  for(var i = 0; i < elements.length; ++i) {
    elements[i].addEventListener('click', rippleEffect, false);
  }
};

function renderMath() {
  var currentEquation = 1;

  // Render all the math-elements.
  var elements = document.getElementsByClassName("math");
  for(var i = 0; i < elements.length; ++i) {
    var e = elements[i];
    var tex = e.innerHTML;
    katex.render(tex, e);
    // Check if the element is an equation.
    if( e.classList.contains("equation-math") ) {
      // Set the unique id of the equation.
      e.id = "equation-" + currentEquation;
      // Add the equation number.
      e.innerHTML += '<span class="equation-math-number">(' + currentEquation + ')</span>';
      ++currentEquation;
    }
  }
};

addRippleEffects();
renderMath();
