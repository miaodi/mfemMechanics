/*
 CircularBuffer.tpp - Circular buffer library for Arduino.
 Copyright (c) 2017 Roberto Lo Giacco.

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

template <typename T, size_t S>
constexpr CircularBuffer<T, S>::CircularBuffer() : head( buffer ), tail( buffer ), count( 0 )
{
}

template <typename T, size_t S>
bool CircularBuffer<T, S>::unshift( const T& value )
{
    if ( head == buffer )
    {
        head = buffer + capacity;
    }
    *--head = value;
    if ( count == capacity )
    {
        if ( tail-- == buffer )
        {
            tail = buffer + capacity - 1;
        }
        return false;
    }
    else
    {
        if ( count++ == 0 )
        {
            tail = head;
        }
        return true;
    }
}

template <typename T, size_t S>
bool CircularBuffer<T, S>::push( const T& value )
{
    if ( ++tail == buffer + capacity )
    {
        tail = buffer;
    }
    *tail = value;
    if ( count == capacity )
    {
        if ( ++head == buffer + capacity )
        {
            head = buffer;
        }
        return false;
    }
    else
    {
        if ( count++ == 0 )
        {
            head = tail;
        }
        return true;
    }
}

template <typename T, size_t S>
const T& CircularBuffer<T, S>::shift()
{
    if ( count == 0 )
        return *head;
    const T& result = *head++;
    if ( head >= buffer + capacity )
    {
        head = buffer;
    }
    count--;
    return result;
}

template <typename T, size_t S>
const T& CircularBuffer<T, S>::pop()
{
    if ( count == 0 )
        return *tail;
    const T& result = *tail--;
    if ( tail < buffer )
    {
        tail = buffer + capacity - 1;
    }
    count--;
    return result;
}

template <typename T, size_t S>
const T& CircularBuffer<T, S>::first() const
{
    return *head;
}

template <typename T, size_t S>
const T& CircularBuffer<T, S>::last() const
{
    return *tail;
}

template <typename T, size_t S>
const T& CircularBuffer<T, S>::operator[]( size_t index ) const
{
    if ( index >= count )
        return *tail;
    return *( buffer + ( ( head - buffer + index ) % capacity ) );
}

template <typename T, size_t S>
T& CircularBuffer<T, S>::operator[]( size_t index )
{
    if ( index >= count )
        return *tail;
    return *( buffer + ( ( head - buffer + index ) % capacity ) );
}

template <typename T, size_t S>
size_t CircularBuffer<T, S>::size() const
{
    return count;
}

template <typename T, size_t S>
size_t CircularBuffer<T, S>::available() const
{
    return capacity - count;
}

template <typename T, size_t S>
bool CircularBuffer<T, S>::isEmpty() const
{
    return count == 0;
}

template <typename T, size_t S>
bool CircularBuffer<T, S>::isFull() const
{
    return count == capacity;
}

template <typename T, size_t S>
void CircularBuffer<T, S>::clear()
{
    head = tail = buffer;
    count = 0;
}

template <typename T, size_t S>
void CircularBuffer<T, S>::copyToArray( T* dest ) const
{
    const T* finish = dest + count;
    for ( const T* current = head; current < ( buffer + capacity ) && dest < finish; current++, dest++ )
    {
        *dest = *current;
    }
    for ( const T* current = buffer; current <= tail && dest < finish; current++, dest++ )
    {
        *dest = *current;
    }
}

template <typename T, size_t S>
template <typename R>
void CircularBuffer<T, S>::copyToArray( R* dest, R ( &convertFn )( const T& ) ) const
{
    const R* finish = dest + count;
    for ( const T* current = head; current < ( buffer + capacity ) && dest < finish; current++, dest++ )
    {
        *dest = convertFn( *current );
    }
    for ( const T* current = buffer; current <= tail && dest < finish; current++, dest++ )
    {
        *dest = convertFn( *current );
    }
}